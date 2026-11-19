#!/usr/bin/env python
"""diag_error_accum.py —— 量化 SSM 的误差累积（μ_z 沿 z 的漂移）。

病根假设（thesis 的"误差累积"）：递归里每一步带一点偏差，μ_{z-1} 传到这一片时
已经漂了（误差累积），长程越传越偏。

方法：对 held-out 序列，跑三种递归模式，取 μ_z（先验），量它相对干净真值 X_clean
的非金属区 RMSE，看偏差是否随递归方向增长：
  forward    单向（误差从顶往下累积）
  backward   单向（误差从底往上累积）
  per_slice  冷启动（无递归，每片独立，无累积 = 基准）

关键量：dev(z) = RMSE(μ_z, X_clean_z)；「递归额外误差」= 单向 dev − 冷启动 dev。
若单向的 dev 随 z 单调增大、且额外误差随相对位置增长 → 实锤误差累积。
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/prism_framework")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prism.data.dataset import make_theta_sequence_loader  # noqa: E402
from prism.models.m_network import MNetwork  # noqa: E402
from prism.models.find_net import FINDNet, FINDNetArgs  # noqa: E402
from prism.recursion.algorithm1 import forward_backward  # noqa: E402
from prism.training.trainer import SSMTrainer  # noqa: E402
from prism.utils.config import SSMConfig  # noqa: E402


def _sample_idx(sid):
    return int(sid.rsplit("_", 1)[1])


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="/root/autodl-tmp/PRISM/training_data_astra/ablations_v13/00_baseline.pt")
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--patient", default="patients_head2")
    ap.add_argument("--max-implants", type=int, default=0)
    ap.add_argument("--out", default="/root/autodl-tmp/PRISM/training_data_astra/viz_v12/diag_error_accum.png")
    args = ap.parse_args()

    dev = "cuda"
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    cfg = ckpt["config"]
    d_ref = cfg["d_ref"]

    m_net = MNetwork(nb=cfg.get("nb", 1), channels=cfg.get("channels", 32),
                     use_velocity=cfg.get("use_velocity", True)).to(dev)
    find_net = FINDNet(FINDNetArgs(use_checkpoint=True)).to(dev)
    m_net.load_state_dict(ckpt["model"])
    find_net.load_state_dict(ckpt["find_net"])
    m_net.eval()
    find_net.eval()

    config = SSMConfig(nb=cfg.get("nb", 1), channels=cfg.get("channels", 32),
                       d_ref=d_ref, gain_mode=cfg.get("gain_mode", "kalman"),
                       findnet_lict_source=cfg.get("findnet_lict_source", "mu"),
                       backfill_r=cfg.get("backfill_r", 8))
    trainer = SSMTrainer(m_net, config, device=dev, find_net=find_net)

    # held-out 序列
    meta_dir = os.path.join(args.data_dir, "meta")
    by_seq = {}
    for f in glob.glob(os.path.join(meta_dir, "*.json")):
        m = json.load(open(f))
        if m.get("patient_id") == args.patient:
            by_seq.setdefault(m["sequence_id"], []).append(m["sample_id"])
    specs = [sorted(v, key=_sample_idx) for v in by_seq.values() if len(v) >= 3]
    if args.max_implants > 0:
        specs = specs[: args.max_implants]
    print(f"[diag] {args.patient}: {len(specs)} 序列", flush=True)

    train_dir = os.path.join(args.data_dir, "train")
    loader = make_theta_sequence_loader(
        os.path.join(train_dir, "Target"), os.path.join(train_dir, "Baseline"),
        os.path.join(train_dir, "LI"), os.path.join(train_dir, "Mask"),
        os.path.join(train_dir, "Phys"))

    # 相对位置 → 各模式的偏差平方累计（用中位片归一相对位置）
    acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))  # mode -> relpos -> [sse, cnt]

    for i, spec in enumerate(specs):
        sample = loader(spec)
        Z = len(spec)
        thetas = [{k: v.unsqueeze(0).to(dev) for k, v in t.items()} for t in sample["thetas"]]
        X_clean = sample["X_clean"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(dev)  # [1,1,Z,H,W]
        metal_mask = sample["metal_mask"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(dev)
        x_hat = sample["x_hat"].unsqueeze(0).to(dev)
        li = sample["li"].unsqueeze(0).to(dev)

        dirty_vol = x_hat.permute(0, 2, 1, 3, 4)
        dirty = [dirty_vol[:, :, z] for z in range(Z)]
        li_vol = li.permute(0, 2, 1, 3, 4)
        li_list = [li_vol[:, :, z] for z in range(Z)]
        masks = [metal_mask[:, :, z] for z in range(Z)]
        observer = trainer._observer(dirty, masks, li_list)

        # 非金属有效掩膜（排除回填 8px）
        binm = (metal_mask > 0.5).float()
        bk = 2 * 8 + 1
        backfill = torch.nn.functional.max_pool3d(binm, kernel_size=(1, bk, bk), stride=1,
                                                  padding=(0, 8, 8))
        valid = (1.0 - backfill).float() * (1.0 - metal_mask)  # [1,1,Z,H,W]

        with torch.no_grad():
            res = forward_backward(m_net, thetas, d_ref, recursion="bidirectional",
                                   observer=observer, gain_mode=config.gain_mode)

        # 三种模式的 μ_z
        modes = {
            "forward": res.fwd,       # [Z] KalmanState
            "backward": res.bwd,
            "fused": res.fused,
        }
        # 冷启动（per_slice）：复用 forward_backward 的 per_slice
        with torch.no_grad():
            res_cold = forward_backward(m_net, thetas, d_ref, recursion="per_slice",
                                        observer=observer, gain_mode=config.gain_mode)
        modes["cold"] = res_cold.fwd

        for mode, states in modes.items():
            for z in range(Z):
                mu_z = states[z].mu  # [1,1,H,W]
                xc = X_clean[:, :, z]  # [1,1,H,W]
                v = valid[:, :, z]     # [1,1,H,W]
                se = ((mu_z - xc) ** 2 * v).sum().item()
                cnt = v.sum().item()
                if cnt == 0:
                    continue
                rel = z / (Z - 1) if Z > 1 else 0.0
                acc[mode][round(rel, 2)][0] += se
                acc[mode][round(rel, 2)][1] += cnt

        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(specs)}", flush=True)

    # 汇总：每模式每相对位置的 RMSE
    modes_order = ["cold", "forward", "backward", "fused"]
    print("\n=== μ_z 相对 X_clean 的 RMSE 随相对位置（0=顶，1=底）===", flush=True)
    curves = {}
    for mode in modes_order:
        rels = sorted(acc[mode].keys())
        rmse = [np.sqrt(acc[mode][r][0] / acc[mode][r][1]) for r in rels]
        curves[mode] = (rels, rmse)
        trend = "".join("↑" if (i and rmse[i] > rmse[i - 1]) else "↓" if (i and rmse[i] < rmse[i - 1]) else "="
                        for i in range(len(rmse)))
        print(f"  {mode:9s}: " + " ".join(f"{r:.2f}:{v:.0f}" for r, v in zip(rels, rmse)) + f"  [{trend}]",
              flush=True)

    # 递归额外误差 = forward/backward − cold
    print("\n=== 递归额外误差（forward−cold / backward−cold）===", flush=True)
    if "cold" in curves:
        cr = dict(zip(curves["cold"][0], curves["cold"][1]))
        for mode in ["forward", "backward"]:
            extra = []
            for r, v in zip(*curves[mode]):
                c = cr.get(r, 0.0)
                extra.append(v - c)
            print(f"  {mode:9s}: " + " ".join(f"{r:.2f}:{e:+.0f}" for r, e in zip(curves[mode][0], extra)),
                  flush=True)

    # 图
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"cold": "gray", "forward": "red", "backward": "blue", "fused": "purple"}
    for mode in modes_order:
        rels, rmse = curves[mode]
        ax.plot(rels, rmse, "o-", color=colors[mode], label=mode, markersize=4)
    ax.set_xlabel("relative z position (0=top, 1=bottom)")
    ax.set_ylabel("RMSE(μ_z, X_clean) [HU]")
    ax.set_title("SSM error accumulation along z (held-out patients_head2)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {args.out}", flush=True)


if __name__ == "__main__":
    main()
