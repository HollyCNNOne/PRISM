#!/usr/bin/env python
"""eval_zcontinuity.py —— 诊断 SSM 隐状态 μ 的 Z 轴连续性（B6/B7 塌缩定位）。

核心假说：μ 不是平的，但塌成了「Z 不变的群体先验」——只学到跨片粗结构，
没学到「这一片的真实解剖」。本脚本量化两个判据（均在非金属体素 I>0.5 上）：

1. 逐片 μ_corr：μ_z 与 X_z 逐片 Pearson 相关（对照 pooled 相关）。
   —— pooled 高、逐片均值低 → μ 只跟跨片粗结构相关，不跟逐片解剖相关。
2. Z-差分比：E|μ_z − μ_{z-1}| / E|X_z − X_{z-1}|。
   —— ≪1 → μ 沿 Z 几乎不移动 = Z 不变塌缩的铁证。
   同时给 baseline x̂ 的同一比值做参照（伪影应使 baseline 比值 >1）。
   并按阴影分区（部分/完全）看 μ 具体在哪里冻结。

开环忠实：forward_backward 传 x_hats=None，复现 checkpoint 训练时的开环 μ
（B6/B7 之前的旧 checkpoint 都是开环训的，注入观测会偏离训练态）。

用法：
    python scripts/eval_zcontinuity.py \
        --checkpoint <ckpt> --data-dir ... --patient patients_head2 \
        --trust-mode kalman --recursion bidirectional --nb 1 --channels 32 --device cuda
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
from prism.recursion.algorithm1 import forward_backward  # noqa: E402


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


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double(); b = b.double()
    ca = a - a.mean(); cb = b - b.mean()
    den = (ca.square().sum() * cb.square().sum()).sqrt()
    return float((ca * cb).sum() / den) if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data")
    ap.add_argument("--patient", required=True)
    ap.add_argument("--trust-mode", default="kalman", choices=["kalman", "gated", "fixed"])
    ap.add_argument("--recursion", default="bidirectional",
                    choices=["bidirectional", "forward", "per_slice"])
    ap.add_argument("--nb", type=int, default=1)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-z", type=int, default=3)
    ap.add_argument("--max-z", type=int, default=0)
    ap.add_argument("--max-implants", type=int, default=0)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = ckpt["config"]
    d_ref = cfg["d_ref"]
    nb = cfg.get("nb", args.nb)
    channels = cfg.get("channels", args.channels)
    use_velocity = cfg.get("use_velocity", True)
    gate_reads_dj = cfg.get("gate_reads_dj", False)

    m_net = MNetwork(nb=nb, channels=channels, use_velocity=use_velocity,
                     gate_reads_dj=gate_reads_dj).to(args.device)
    m_net.load_state_dict(ckpt["model"])
    m_net.eval()

    print(f"[zcont] checkpoint={args.checkpoint}\n"
          f"  patient={args.patient!r} recursion={args.recursion} "
          f"use_velocity={use_velocity} gate_reads_dj={gate_reads_dj} "
          f"d_ref={d_ref:.6g}", flush=True)

    train_dir = os.path.join(args.data_dir, "train")
    meta_dir = os.path.join(args.data_dir, "meta")
    specs, metas = build_patient_specs(meta_dir, args.patient,
                                       min_z=args.min_z, max_z=args.max_z)
    if args.max_implants > 0:
        specs = specs[: args.max_implants]
    print(f"[zcont] {len(specs)} 序列", flush=True)

    loader = make_theta_sequence_loader(
        os.path.join(train_dir, "Target"), os.path.join(train_dir, "Baseline"),
        os.path.join(train_dir, "LI"), os.path.join(train_dir, "Mask"),
        os.path.join(train_dir, "Phys"))

    # 逐片 corr 列表（每片一个值）
    slice_corrs: list[float] = []
    # pooled corr 运行和（对照）
    pSmu = pSX = pSmm = pSXX = pSmX = 0.0
    pN = 0
    # Z 差分运行和：总体 + 分区（部分阴影 / 完全阴影）
    acc = {"总体": [0.0, 0.0, 0.0, 0],   # [S_dmu, S_dX, S_db, N_pairs]
           "部分阴影": [0.0, 0.0, 0.0, 0],
           "完全阴影": [0.0, 0.0, 0.0, 0]}

    with torch.no_grad():
        for i, spec in enumerate(specs):
            sample = loader(spec)
            thetas = [{k: v.unsqueeze(0).to(args.device) for k, v in t.items()}
                      for t in sample["thetas"]]
            x_hat = sample["x_hat"].unsqueeze(0).to(args.device)
            X_clean = sample["X_clean"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(args.device)
            metal_mask = sample["metal_mask"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(args.device)

            res = forward_backward(m_net, thetas, d_ref, recursion=args.recursion,
                                   x_hats=None)          # 开环，忠实训练态
            fused = res[2]
            mu = torch.stack([s.mu for s in fused], dim=2)   # [1,1,Z,H,W]
            xb = x_hat.permute(0, 2, 1, 3, 4)                # [1,1,Z,H,W] baseline
            d_j = torch.stack([t["d_j"] for t in thetas], dim=2)  # [1,1,Z,H,W]

            I = (1.0 - metal_mask) > 0.5                      # 非金属
            r = d_j / d_ref
            partial = (r >= 1.0 / 9.0) & (r <= 9.0)
            complete = r < 1.0 / 9.0

            Z = mu.shape[2]

            # 逐片 corr + pooled corr 运行和
            for z in range(Z):
                sel = I[0, 0, z]
                if int(sel.sum()) < 100:
                    continue
                mv = mu[0, 0, z][sel]; Xv = X_clean[0, 0, z][sel]
                c = _pearson(mv, Xv)
                if not np.isnan(c):
                    slice_corrs.append(c)
                # pooled 运行和（同一批体素）
                m = mv.double().sum(); X = Xv.double().sum()
                pSmu += float(m); pSX += float(X)
                pSmm += float((mv.double() * mv.double()).sum())
                pSXX += float((Xv.double() * Xv.double()).sum())
                pSmX += float((mv.double() * Xv.double()).sum())
                pN += int(sel.sum())

            # Z 差分（逐相邻片对）
            for z in range(1, Z):
                sel = I[0, 0, z] & I[0, 0, z - 1]
                if int(sel.sum()) < 100:
                    continue
                dmu = float((mu[0, 0, z] - mu[0, 0, z - 1]).abs()[sel].mean())
                dX = float((X_clean[0, 0, z] - X_clean[0, 0, z - 1]).abs()[sel].mean())
                db = float((xb[0, 0, z] - xb[0, 0, z - 1]).abs()[sel].mean())
                for name, sela in (("总体", sel), ("部分阴影", sel & partial[0, 0, z]),
                                   ("完全阴影", sel & complete[0, 0, z])):
                    if int(sela.sum()) < 100:
                        continue
                    dmu_s = float((mu[0, 0, z] - mu[0, 0, z - 1]).abs()[sela].mean())
                    dX_s = float((X_clean[0, 0, z] - X_clean[0, 0, z - 1]).abs()[sela].mean())
                    db_s = float((xb[0, 0, z] - xb[0, 0, z - 1]).abs()[sela].mean())
                    a = acc[name]
                    a[0] += dmu_s; a[1] += dX_s; a[2] += db_s; a[3] += 1

            if (i + 1) % 25 == 0:
                print(f"  ... {i + 1}/{len(specs)} 序列", flush=True)

    # pooled corr
    cov = pSmX - pSmu * pSX / pN
    vmu = pSmm - pSmu ** 2 / pN
    vX = pSXX - pSX ** 2 / pN
    pooled_corr = float(cov / np.sqrt(vmu * vX)) if (vmu > 0 and vX > 0) else float("nan")

    sc = np.array(slice_corrs)
    print("\n[zcont] ==== μ 的 Z 轴连续性诊断 ====", flush=True)
    print(f"[1] pooled μ_corr          = {pooled_corr:.4f}  (跨片粗结构相关)", flush=True)
    print(f"    逐片 μ_corr: n={len(sc)}  mean={sc.mean():.4f}  std={sc.std():.4f}  "
          f"median={np.median(sc):.4f}  P(<0.5)={(sc < 0.5).mean():.3f}  "
          f"P(<0.3)={(sc < 0.3).mean():.3f}", flush=True)
    print(f"\n[2] Z-差分比  E|μ_z−μ_{z-1}| / E|X_z−X_{z-1}|   （1≈追踪解剖，≪1=Z 不变塌缩）", flush=True)
    print(f"{'分区':<10} {'μ差分/X差分':>14} {'baseline差分/X差分':>20} {'片对数':>8}", flush=True)
    for name in ("总体", "部分阴影", "完全阴影"):
        S_dmu, S_dX, S_db, N = acc[name]
        if N == 0:
            continue
        ratio_mu = S_dmu / S_dX
        ratio_b = S_db / S_dX
        print(f"{name:<10} {ratio_mu:>14.4f} {ratio_b:>20.4f} {N:>8}", flush=True)


if __name__ == "__main__":
    main()
