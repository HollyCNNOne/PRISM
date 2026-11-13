#!/usr/bin/env python
"""eval_knob_sweep.py —— C 消融（H2）的「手调旋钮 / 可学习参数」敏感性扫描（eval-only，不重训）。

【已废弃】递归循环定案（prism-interface-loss-final）后，信任通道 π=g⊙c 与 R_θ 从
train/eval 路径移除（信任隐含在 μ_pred/K 里，最终输出 = FIND-Net 唯一输出 x̂_z）。
本脚本的「旋钮→trust→R_θ」命题在新设计下不再存在，仅保留作 H2 定案档案；新 checkpoint
存 `find_net`（非 `r_theta`），本脚本只适配旧 checkpoint。

针对已训好的 gated checkpoint（C1），冻结 MNetwork + c_head + R_θ 全部权重，只改
g(d_j) 的手调旋钮（d_ref_gate = d_ref_rel·d_ref、指数 k）并可选关掉 c_head 关联
（π = g 无学习关联），对 held-out 病人重跑三分区 RMSE / PSNR。

回答的问题（H2 的「0 旋钮 0 参数」是鲁棒性/正确性命题，不是效率命题）：
    1. 手调旋钮的代价：把 d_ref_gate 从 0.01×d_ref（旧透明重建调的值）扫到 1×d_ref
       （Fisher 信息的天然尺度），看部分阴影区 RMSE 塌多少 —— 旋钮调错 = 静默掉 40%。
    2. 学习 c_head 的代价：关掉 c（π=g 无学习关联），看那 641 参数是不是 load-bearing 的拐杖。
    3. 参考行：同一模型把信任信号换成卡尔曼后验总精度 p_post_sum（0 旋钮 0 参数），对照。

SSM 递归（forward_backward）与旋钮无关 → 每体只展开一次，各设置只重算 trust + R_θ 前向，
开销 ≈ 一次 eval（不是 N 次）。

用法：
    python scripts/eval_knob_sweep.py \
        --checkpoint .../ssm_nb1_gated_ho_patients_head2_200ep.pt \
        --data-dir /root/autodl-tmp/PRISM/training_data \
        --patient patients_head2 --nb 1 --channels 32 --device cuda
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

_FRAMEWORK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _FRAMEWORK_ROOT)

from prism.data.dataset import make_theta_sequence_loader  # noqa: E402
from prism.models.m_network import MNetwork  # noqa: E402
from prism.models.r_theta import RTheta  # noqa: E402
from prism.recursion.algorithm1 import forward_backward  # noqa: E402
from prism.kalman.state import total_precision  # noqa: E402
from prism.training.trainer import SSMTrainer  # noqa: E402
from prism.utils.config import SSMConfig  # noqa: E402


def _sample_idx(sample_id: str) -> int:
    return int(sample_id.rsplit("_", 1)[1])


def build_patient_specs(meta_dir: str, patient_id: str, min_z: int = 3, max_z: int = 0):
    metas = []
    for f in glob.glob(os.path.join(meta_dir, "*.json")):
        m = json.load(open(f))
        if m.get("patient_id") == patient_id:
            metas.append(m)
    by_seq = {}
    for m in metas:
        by_seq.setdefault(m["sequence_id"], []).append(m["sample_id"])
    specs = [sorted(sids, key=_sample_idx) for sids in by_seq.values()]
    specs = [s for s in specs if len(s) >= min_z]
    if max_z > 0:
        specs = [s[:max_z] for s in specs]
    specs.sort(key=lambda s: (_sample_idx(s[0]), len(s)))
    return specs, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data")
    ap.add_argument("--patient", required=True)
    ap.add_argument("--nb", type=int, default=1)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-z", type=int, default=3)
    ap.add_argument("--max-z", type=int, default=0)
    ap.add_argument("--max-implants", type=int, default=0)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = ckpt["config"]
    d_ref = cfg["d_ref"]
    nb = cfg.get("nb", args.nb)
    channels = cfg.get("channels", args.channels)

    m_net = MNetwork(nb=nb, channels=channels).to(args.device)
    r_theta = RTheta(channels=channels).to(args.device)
    m_net.load_state_dict(ckpt["model"])
    r_theta.load_state_dict(ckpt["r_theta"])
    m_net.eval()
    r_theta.eval()

    # trust_mode 固定 gated（本脚本只扫 gated 的旋钮/关联），config 仅给 d_ref/scale 用。
    config = SSMConfig(nb=nb, channels=channels, trust_mode="gated", d_ref=d_ref)
    trainer = SSMTrainer(m_net, config, device=args.device, r_theta=r_theta)

    print(f"[sweep] checkpoint={args.checkpoint}\n"
          f"  patient={args.patient!r}  nb={nb}  d_ref={d_ref:.6g}", flush=True)

    train_dir = os.path.join(args.data_dir, "train")
    meta_dir = os.path.join(args.data_dir, "meta")
    specs, metas = build_patient_specs(meta_dir, args.patient,
                                        min_z=args.min_z, max_z=args.max_z)
    if args.max_implants > 0:
        specs = specs[: args.max_implants]
    print(f"[sweep] {args.patient!r}: {len(specs)} 序列 / {len(metas)} 样本", flush=True)

    loader = make_theta_sequence_loader(
        os.path.join(train_dir, "Target"), os.path.join(train_dir, "Baseline"),
        os.path.join(train_dir, "LI"), os.path.join(train_dir, "Mask"),
        os.path.join(train_dir, "Phys"))

    # (name, kind, kwargs) —— kind ∈ {gated, kalman_ref}；gated 的 kwargs 传给 compute_gated_trust。
    settings = [
        ("rel=0.01 k=3 (原)", "gated", {"d_ref_rel": 0.01, "k": 3.0, "use_c": True}),
        ("rel=0.1  k=3",      "gated", {"d_ref_rel": 0.1,  "k": 3.0, "use_c": True}),
        ("rel=1.0  k=3 (kalman尺度)", "gated", {"d_ref_rel": 1.0, "k": 3.0, "use_c": True}),
        ("rel=0.01 k=1",      "gated", {"d_ref_rel": 0.01, "k": 1.0, "use_c": True}),
        ("rel=1.0  k=1",      "gated", {"d_ref_rel": 1.0,  "k": 1.0, "use_c": True}),
        ("rel=0.01 k=3 无c",  "gated", {"d_ref_rel": 0.01, "k": 3.0, "use_c": False}),
        ("rel=1.0  k=3 无c",  "gated", {"d_ref_rel": 1.0,  "k": 3.0, "use_c": False}),
        ("kalman p_post_sum (参考,0旋钮0参数)", "kalman_ref", {}),
    ]

    # 每个设置的三分区累计（非金属区）
    SSE = {name: {"充分": 0.0, "部分阴影": 0.0, "完全阴影": 0.0} for name, _, _ in settings}
    CNT = {name: {"充分": 0, "部分阴影": 0, "完全阴影": 0} for name, _, _ in settings}
    peak = 0.0

    with torch.no_grad():
        for i, spec in enumerate(specs):
            sample = loader(spec)
            thetas = [{k: v.unsqueeze(0).to(args.device) for k, v in t.items()}
                      for t in sample["thetas"]]
            x_hat = sample["x_hat"].unsqueeze(0).to(args.device)            # [1,Z,1,H,W]
            X_clean = sample["X_clean"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(args.device)
            metal_mask = sample["metal_mask"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(args.device)

            # SSM 递归与旋钮无关 → 每体只展开一次
            (_, _, fused, _, _, _, _, _, _) = forward_backward(m_net, thetas, d_ref)
            mu_fused = trainer._stack([s.mu for s in fused])            # [1,1,Z,H,W]
            p_post_fused = trainer._stack([s.p_post for s in fused])    # [1,nb,Z,H,W]
            d_j = trainer._stack([t["d_j"] for t in thetas])            # [1,1,Z,H,W]

            I = 1.0 - metal_mask
            r = d_j / d_ref
            peak = max(peak, float(X_clean[I > 0.5].max().item()))

            for name, kind, kwargs in settings:
                if kind == "kalman_ref":
                    trust = total_precision(p_post_fused)               # [1,1,Z,H,W]
                else:
                    _, _, Z, _, _ = mu_fused.shape
                    trust = torch.stack([
                        m_net.compute_gated_trust(
                            mu_fused[:, :, z, :, :], d_j[:, :, z, :, :], d_ref, **kwargs)
                        for z in range(Z)], dim=2)                      # [1,1,Z,H,W]

                x_hat_final = r_theta(x_hat.permute(0, 2, 1, 3, 4), mu_fused, trust,
                                      d_j, d_ref)
                err = (x_hat_final - X_clean) ** 2

                for zone in SSE[name]:
                    sel = I > 0.5
                    if zone == "充分":
                        sel = sel & (r > 9.0)
                    elif zone == "完全阴影":
                        sel = sel & (r < 1.0 / 9.0)
                    else:  # 部分阴影
                        sel = sel & (r >= 1.0 / 9.0) & (r <= 9.0)
                    SSE[name][zone] += float(err[sel].sum().item())
                    CNT[name][zone] += int(sel.sum().item())

            if (i + 1) % 25 == 0:
                print(f"  ... {i + 1}/{len(specs)} 序列", flush=True)

    # 报告
    print(f"\n[结果] peak(HU)={peak:.1f}\n", flush=True)
    hdr = f"{'设置':<30} {'部分阴影RMSE':>14} {'完全阴影RMSE':>14} {'充分RMSE':>12}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    out = {}
    for name, kind, kwargs in settings:
        row = {"kind": kind, "kwargs": kwargs}
        for zone in SSE[name]:
            cnt = CNT[name][zone]
            rmse = float(np.sqrt(SSE[name][zone] / cnt)) if cnt > 0 else float("nan")
            row[zone] = {"n_voxels": cnt, "rmse": rmse}
        out[name] = row
        print(f"{name:<30} {row['部分阴影']['rmse']:>14.3f} "
              f"{row['完全阴影']['rmse']:>14.3f} {row['充分']['rmse']:>12.3f}", flush=True)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"patient": args.patient, "d_ref": d_ref, "peak_hu": peak,
                       "results": out, "n_implants": len(specs)}, f, indent=2)
        print(f"\n[save] {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
