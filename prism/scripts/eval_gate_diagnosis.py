#!/usr/bin/env python
"""eval_gate_diagnosis.py —— 定位「μ 过平滑」的源头：Δ 太小？K 太软？还是 M_μ 本身糊？

背景（zcontinuity 已定案）：μ 不是塌缩，而是「沿 Z 被低通滤波的解剖」
（Z-差分比 μ=0.35 vs baseline=2.53）。本脚本把平滑源头拆成三个可测的量：

1. Δ（预测步门控）分布：Δ→0 = 冻结 carry（把上一片 μ 原样带过来 = 平滑）。
   期望：部分阴影 Δ→0.9（更新到当前片、保细节），只在完全阴影 Δ→0。
2. K（卡尔曼增益，信先验权重）分布：K = p_sum/(p_sum + d_j/d_ref)。
   期望：部分阴影 K→0（信观测 x̂），完全阴影 K→1（信先验、别把打穿的观测漏进 μ）。
   B7 完全阴影 μ_bias +686 的嫌疑 = K 在完全阴影不够尖（p_sum 太小）。
3. M_μ 冷启动预测本身的 Z-差分比（per_slice 模式 μ = M_μ(S0, Θ_z) 无 carry）：
   若它也 ≈0.35 → 糊的是 M_μ 预测网络；若 ≈0.8 → 糊的是递归 Δ，M_μ 无辜。

均统计在非金属体素（I>0.5），按 d_j/d_ref 分「部分阴影 / 完全阴影」。

用法：
    python scripts/eval_gate_diagnosis.py \
        --checkpoint <ckpt> --data-dir ... --patient patients_head2 \
        --nb 1 --channels 32 --device cuda
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

    print(f"[gate] checkpoint={args.checkpoint}\n"
          f"  patient={args.patient!r} use_velocity={use_velocity} "
          f"gate_reads_dj={gate_reads_dj} d_ref={d_ref:.6g}", flush=True)

    train_dir = os.path.join(args.data_dir, "train")
    meta_dir = os.path.join(args.data_dir, "meta")
    specs, metas = build_patient_specs(meta_dir, args.patient,
                                       min_z=args.min_z, max_z=args.max_z)
    if args.max_implants > 0:
        specs = specs[: args.max_implants]
    print(f"[gate] {len(specs)} 序列", flush=True)

    loader = make_theta_sequence_loader(
        os.path.join(train_dir, "Target"), os.path.join(train_dir, "Baseline"),
        os.path.join(train_dir, "LI"), os.path.join(train_dir, "Mask"),
        os.path.join(train_dir, "Phys"))

    # Δ / K 累加器：每个 zone 存 (sum, n)
    dsum = {"部分阴影": [0.0, 0], "完全阴影": [0.0, 0], "全部": [0.0, 0]}
    ksum = {"部分阴影": [0.0, 0], "完全阴影": [0.0, 0], "全部": [0.0, 0]}
    # M_μ 冷启动 Z-差分比运行和
    S_dmu = S_dX = 0.0
    N_pairs = 0

    with torch.no_grad():
        for i, spec in enumerate(specs):
            sample = loader(spec)
            thetas = [{k: v.unsqueeze(0).to(args.device) for k, v in t.items()}
                      for t in sample["thetas"]]
            X_clean = sample["X_clean"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(args.device)
            metal_mask = sample["metal_mask"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(args.device)

            d_j = torch.stack([t["d_j"] for t in thetas], dim=2)   # [1,1,Z,H,W]
            r = d_j / d_ref
            I = (1.0 - metal_mask) > 0.5
            partial = (r >= 1.0 / 9.0) & (r <= 9.0)
            complete = r < 1.0 / 9.0

            # 双向递归 → 取 Δ 和 K（前向趟；后向趟同分布，只报前向）
            res = forward_backward(m_net, thetas, d_ref, recursion="bidirectional",
                                   x_hats=None)
            delta = torch.stack(res[7], dim=2)   # fwd_delta [1,1,Z,H,W]
            K = torch.stack(res[3], dim=2)       # fwd_gains [1,1,Z,H,W]

            for name, zone in (("部分阴影", partial), ("完全阴影", complete),
                               ("全部", I)):
                sel = I & zone
                if int(sel.sum()) == 0:
                    continue
                dsum[name][0] += float(delta[sel].sum()); dsum[name][1] += int(sel.sum())
                ksum[name][0] += float(K[sel].sum()); ksum[name][1] += int(sel.sum())

            # M_μ 冷启动预测（per_slice，无 carry）：μ = M_μ(S0, Θ_z)
            ps = forward_backward(m_net, thetas, d_ref, recursion="per_slice",
                                  x_hats=None)[2]
            mu_cold = torch.stack([s.mu for s in ps], dim=2)  # [1,1,Z,H,W]
            Z = mu_cold.shape[2]
            for z in range(1, Z):
                sel = I[0, 0, z] & I[0, 0, z - 1]
                if int(sel.sum()) < 100:
                    continue
                S_dmu += float((mu_cold[0, 0, z] - mu_cold[0, 0, z - 1]).abs()[sel].mean())
                S_dX += float((X_clean[0, 0, z] - X_clean[0, 0, z - 1]).abs()[sel].mean())
                N_pairs += 1

            if (i + 1) % 25 == 0:
                print(f"  ... {i + 1}/{len(specs)} 序列", flush=True)

    print("\n[gate] ==== 平滑源头定位 ====", flush=True)
    print(f"[1] Δ（预测步门控，→0=冻结 carry）均值：", flush=True)
    for name in ("全部", "部分阴影", "完全阴影"):
        s, n = dsum[name]
        print(f"    {name:<8} Δ = {s / n:.4f}   (n={n})", flush=True)
    print(f"\n[2] K（卡尔曼增益，→0=信观测，→1=信先验）均值：", flush=True)
    for name in ("全部", "部分阴影", "完全阴影"):
        s, n = ksum[name]
        print(f"    {name:<8} K = {s / n:.4f}   (n={n})", flush=True)
    print(f"\n[3] M_μ 冷启动预测 Z-差分比（无 carry，纯预测网络）="
          f"{S_dmu / S_dX:.4f}   （对比：双向 μ 该比值=0.35）", flush=True)


if __name__ == "__main__":
    main()
