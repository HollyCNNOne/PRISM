#!/usr/bin/env python
"""eval_size_scaling.py —— B8 跨线圈尺寸（金属严重度）分层评测。

验证论文叙事：「递归先验的主战场是信息严重缺失（完全阴影）体素，增益随金属严重度
（完全阴影占比）单调放大」。对 held-out 病人，按 sample meta 的 `category`（= 线圈尺寸
small/medium/large/giant，3D_Coil 全因子设计写入）分层，对每个尺寸报告：

  - 完全阴影占比 f = 完全阴影非金属体素 / 全部非金属体素
  - baseline（输入 x̂）完全 / 部分阴影 RMSE
  - B8（x̂_final）完全 / 部分阴影 RMSE
  - 完全阴影增益比 = baseline / B8（论文的 3.2×）

不重训：B8 的影子门 s=1{r≥1/9} 是确定性函数（无学习参数）、μ 先验在「尺寸混合」的
训练集上已见过全部尺寸，故一个 checkpoint 跨尺寸 eval 即可。分区边界沿用 eval_mar.py：
充分 r>9 / 部分阴影 r∈[1/9,9] / 完全阴影 r<1/9，d_ref 来自 checkpoint（训练集冻结）。

用法：
    python scripts/eval_size_scaling.py \
        --checkpoint /root/autodl-tmp/PRISM/training_data/ssm_nb1_kalman_ho_patients_head2_b8_shadow.pt \
        --data-dir /root/autodl-tmp/PRISM/training_data \
        --patient patients_head2 --trust-mode kalman --recursion bidirectional \
        --nb 1 --channels 32 --device cuda \
        --out-json /root/autodl-tmp/PRISM/training_data/size_scaling_b8.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

_FRAMEWORK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _FRAMEWORK_ROOT)

from prism.data.dataset import make_theta_sequence_loader  # noqa: E402
from prism.models.m_network import MNetwork  # noqa: E402
from prism.models.find_net import FINDNet, FINDNetArgs  # noqa: E402
from prism.recursion.algorithm1 import forward_backward  # noqa: E402
from prism.training.trainer import SSMTrainer  # noqa: E402
from prism.utils.config import SSMConfig  # noqa: E402

SIZE_ORDER = ["small", "medium", "large", "giant"]


def _sample_idx(sample_id: str) -> int:
    return int(sample_id.rsplit("_", 1)[1])


def build_size_specs(meta_dir: str, patient_id: str, min_z: int = 3, max_z: int = 0):
    """按 category（线圈尺寸）分组：{category: (specs, n_samples)}。"""
    metas_by_cat = defaultdict(list)
    for f in glob.glob(os.path.join(meta_dir, "*.json")):
        m = json.load(open(f))
        if m.get("patient_id") == patient_id:
            metas_by_cat[m.get("category")].append(m)
    out = {}
    for cat, metas in metas_by_cat.items():
        by_seq = {}
        for m in metas:
            by_seq.setdefault(m["sequence_id"], []).append(m["sample_id"])
        specs = [sorted(sids, key=_sample_idx) for sids in by_seq.values()]
        specs = [s for s in specs if len(s) >= min_z]
        if max_z > 0:
            specs = [s[:max_z] for s in specs]
        specs.sort(key=lambda s: (_sample_idx(s[0]), len(s)))
        out[cat] = specs
    return out


def _zone_sel(name, I, r):
    sel = I > 0.5
    if name == "充分":
        sel = sel & (r > 9.0)
    elif name == "完全阴影":
        sel = sel & (r < 1.0 / 9.0)
    elif name == "部分阴影":
        sel = sel & (r >= 1.0 / 9.0) & (r <= 9.0)
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data")
    ap.add_argument("--patient", required=True, help="held-out 病人 patient_id")
    ap.add_argument("--trust-mode", default="kalman", choices=["kalman", "gated", "fixed"])
    ap.add_argument("--recursion", default="bidirectional",
                    choices=["bidirectional", "forward", "per_slice"])
    ap.add_argument("--nb", type=int, default=1)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-z", type=int, default=3)
    ap.add_argument("--max-z", type=int, default=0)
    ap.add_argument("--decouple-mean", action="store_true",
                    help="B9 开环参考：状态 carry 先验 μ_pred（观测不回灌 carry），测完全阴影的「纯先验地板」")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = ckpt["config"]
    d_ref = cfg["d_ref"]
    nb = cfg.get("nb", args.nb)
    channels = cfg.get("channels", args.channels)
    use_velocity = cfg.get("use_velocity", True)
    gate_reads_dj = cfg.get("gate_reads_dj", False)

    m_net = MNetwork(nb=nb, channels=channels, use_velocity=use_velocity,
                     gate_reads_dj=gate_reads_dj,
                     decouple_mean=args.decouple_mean).to(args.device)
    find_net = FINDNet(FINDNetArgs()).to(args.device)
    m_net.load_state_dict(ckpt["model"])
    find_net.load_state_dict(ckpt["find_net"])
    m_net.eval()
    find_net.eval()

    config = SSMConfig(nb=nb, channels=channels, trust_mode=args.trust_mode,
                       recursion=args.recursion, d_ref=d_ref)
    trainer = SSMTrainer(m_net, config, device=args.device, find_net=find_net)

    print(f"[eval] checkpoint={args.checkpoint}\n"
          f"  patient={args.patient!r}  trust_mode={args.trust_mode}  nb={nb}  "
          f"d_ref={d_ref:.6g}  decouple_mean={args.decouple_mean}", flush=True)

    train_dir = os.path.join(args.data_dir, "train")
    meta_dir = os.path.join(args.data_dir, "meta")
    size_specs = build_size_specs(meta_dir, args.patient,
                                  min_z=args.min_z, max_z=args.max_z)

    loader = make_theta_sequence_loader(
        os.path.join(train_dir, "Target"), os.path.join(train_dir, "Baseline"),
        os.path.join(train_dir, "LI"), os.path.join(train_dir, "Mask"),
        os.path.join(train_dir, "Phys"))

    ZONES = ["完全阴影", "部分阴影", "充分"]
    SSE_base = defaultdict(lambda: defaultdict(float))
    SSE_b8 = defaultdict(lambda: defaultdict(float))
    SUM_b8 = defaultdict(lambda: defaultdict(float))
    CNT = defaultdict(lambda: defaultdict(int))
    peak = 0.0

    with torch.no_grad():
        for cat in sorted(size_specs, key=lambda c: SIZE_ORDER.index(c) if c in SIZE_ORDER else 99):
            specs = size_specs[cat]
            for i, spec in enumerate(specs):
                sample = loader(spec)
                thetas = [{k: v.unsqueeze(0).to(args.device) for k, v in t.items()}
                          for t in sample["thetas"]]
                x_hat = sample["x_hat"].unsqueeze(0).to(args.device)      # [1,Z,1,H,W]
                X_clean = sample["X_clean"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(args.device)
                metal_mask = sample["metal_mask"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(args.device)

                Z = len(thetas)
                dirty_vol = x_hat.permute(0, 2, 1, 3, 4)              # [1,1,Z,H,W]
                dirty = [dirty_vol[:, :, z] for z in range(Z)]        # [Z] 个 [1,1,H,W]
                masks = [metal_mask[:, :, z] for z in range(Z)]       # [Z] 个 [1,1,H,W]
                observer = trainer._observer(dirty, masks)
                res = forward_backward(m_net, thetas, d_ref,
                                       recursion=args.recursion, observer=observer)
                d_j = trainer._stack([t["d_j"] for t in thetas])
                x_hat_final = trainer._final_output(res, dirty, masks)

                I = 1.0 - metal_mask
                r = d_j / d_ref
                res_base = x_hat.permute(0, 2, 1, 3, 4) - X_clean
                res_b8 = x_hat_final - X_clean
                err_base = res_base ** 2
                err_b8 = res_b8 ** 2
                peak = max(peak, float(X_clean[I > 0.5].max().item()))

                for name in ZONES:
                    sel = _zone_sel(name, I, r)
                    SSE_base[cat][name] += float(err_base[sel].sum().item())
                    SSE_b8[cat][name] += float(err_b8[sel].sum().item())
                    SUM_b8[cat][name] += float(res_b8[sel].sum().item())
                    CNT[cat][name] += int(sel.sum().item())

    print(f"\n[结果] peak(HU)={peak:.1f}", flush=True)
    header = (f"{'尺寸':<8} {'序列':>5} {'样本':>6} {'完全阴影%':>10} "
              f"{'base完全':>10} {'B8完全':>9} {'比':>6} "
              f"{'base部分':>10} {'B8部分':>9} "
              f"{'B8完全bias':>11} {'B8完全std':>10}")
    print(header, flush=True)
    out = {"patient": args.patient, "trust_mode": args.trust_mode, "d_ref": d_ref,
           "peak_hu": peak, "sizes": {}}
    for cat in SIZE_ORDER:
        if cat not in CNT:
            continue
        n_total = sum(CNT[cat].values())
        n_complete = CNT[cat]["完全阴影"]
        frac = 100.0 * n_complete / n_total if n_total > 0 else float("nan")
        def rmse(acc, name):
            c = CNT[cat][name]
            return float(np.sqrt(acc[cat][name] / c)) if c > 0 else float("nan")
        def bias_mean(name):
            c = CNT[cat][name]
            return float(SUM_b8[cat][name] / c) if c > 0 else float("nan")
        def bias_std(name):
            c = CNT[cat][name]
            if c <= 0:
                return float("nan")
            mu = SUM_b8[cat][name] / c
            var = SSE_b8[cat][name] / c - mu * mu
            return float(np.sqrt(var)) if var > 0 else 0.0
        b_c = rmse(SSE_base, "完全阴影"); m_c = rmse(SSE_b8, "完全阴影")
        b_p = rmse(SSE_base, "部分阴影"); m_p = rmse(SSE_b8, "部分阴影")
        ratio = (b_c / m_c) if (m_c > 0 and b_c > 0) else float("nan")
        bias_c = bias_mean("完全阴影")
        std_c = bias_std("完全阴影")
        n_seq = len(size_specs[cat])
        n_samp = sum(len(s) for s in size_specs[cat])
        print(f"{cat:<8} {n_seq:>5} {n_samp:>6} {frac:>9.2f}% "
              f"{b_c:>10.2f} {m_c:>9.2f} {ratio:>6.2f} "
              f"{b_p:>10.2f} {m_p:>9.2f} "
              f"{bias_c:>11.2f} {std_c:>10.2f}", flush=True)
        out["sizes"][cat] = {
            "n_sequences": n_seq, "n_samples": n_samp,
            "complete_shadow_fraction_pct": frac,
            "complete_shadow_voxels": n_complete,
            "total_voxels": n_total,
            "base_rmse_complete": b_c, "b8_rmse_complete": m_c,
            "ratio_complete": ratio,
            "base_rmse_partial": b_p, "b8_rmse_partial": m_p,
            "b8_complete_bias_hu": bias_c,
            "b8_complete_std_hu": std_c,
        }

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[save] {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
