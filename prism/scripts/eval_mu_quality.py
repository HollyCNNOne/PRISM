#!/usr/bin/env python
"""eval_mu_quality.py —— 诊断「μ 是不是干净的 μ」（B 消融根因判定）。

问题：理论假设递归把「干净组织先验 μ」传进金属遮蔽区；但损失里没有「μ ≈ X_clean」的
直接项（只有 L_down 经 x̂_final=μ+corr 的 ∂/∂μ=1 干净路径 + L_pc 别变金属）。若 μ 其实是
一坨偏置/噪声，递归传什么都是白传 → trust/velocity 全失效。

本脚本对给定 checkpoint，在非金属区按数据充分度三分区（与 eval_mar.py 同边界）直接量：

    μ 对 X_clean 的 RMSE / Pearson 相关 / 偏置   ← 核心判据
    x̂_final 对 X_clean 的 RMSE                   ← 参照（R_θ 最终输出）
    x̂（Baseline 观测）对 X_clean 的 RMSE         ← 参照（无模型下限）

判据：
    * 充分区（r>9）μ 的相关应接近 1、RMSE 小 —— 否则 SSM 连「干净组织先验」都没学会。
    * 阴影区 μ 的 RMSE 在 Full（双向）应明显小于 B3（per_slice）—— 否则递归没在传信息。
      （Full vs B3 用各自 checkpoint + 各自 recursion 各跑一次，对比阴影区 μ 质量。）

用法：
    python scripts/eval_mu_quality.py \
        --checkpoint <ckpt> --data-dir ... --patient patients_head2 \
        --trust-mode kalman --recursion bidirectional \
        --nb 1 --channels 32 --device cuda [--out-json <json>]
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
from prism.models.find_net import FINDNet, FINDNetArgs  # noqa: E402
from prism.recursion.algorithm1 import forward_backward  # noqa: E402
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
    ap.add_argument("--baseline-dir", default=None,
                    help="覆盖观测目录（默认 train/Baseline）；OOD 检查指向 Baseline_bcorr 等")
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
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = ckpt["config"]
    d_ref = cfg["d_ref"]
    nb = cfg.get("nb", args.nb)
    channels = cfg.get("channels", args.channels)
    use_velocity = cfg.get("use_velocity", True)
    gate_reads_dj = cfg.get("gate_reads_dj", False)
    ablate_xm0 = cfg.get("ablate_xm0", False)
    ablate_dj = cfg.get("ablate_dj", False)
    ablate_g = cfg.get("ablate_g", False)

    m_net = MNetwork(nb=nb, channels=channels, use_velocity=use_velocity,
                     gate_reads_dj=gate_reads_dj,
                     ablate_xm0=ablate_xm0, ablate_dj=ablate_dj, ablate_g=ablate_g).to(args.device)
    find_net = FINDNet(FINDNetArgs()).to(args.device)
    m_net.load_state_dict(ckpt["model"])
    find_net.load_state_dict(ckpt["find_net"])
    m_net.eval()
    find_net.eval()

    config = SSMConfig(nb=nb, channels=channels, trust_mode=args.trust_mode,
                       recursion=args.recursion, d_ref=d_ref)
    trainer = SSMTrainer(m_net, config, device=args.device, find_net=find_net)

    train_dir = os.path.join(args.data_dir, "train")
    meta_dir = os.path.join(args.data_dir, "meta")
    specs, metas = build_patient_specs(meta_dir, args.patient,
                                       min_z=args.min_z, max_z=args.max_z)
    if args.max_implants > 0:
        specs = specs[: args.max_implants]
    print(f"[muq] checkpoint={args.checkpoint}\n"
          f"  patient={args.patient!r} trust={args.trust_mode} recursion={args.recursion} "
          f"use_velocity={use_velocity} d_ref={d_ref:.6g}  {len(specs)} 序列", flush=True)

    baseline_dir = os.path.join(train_dir, "Baseline") if not args.baseline_dir else args.baseline_dir
    loader = make_theta_sequence_loader(
        os.path.join(train_dir, "Target"), baseline_dir,
        os.path.join(train_dir, "LI"), os.path.join(train_dir, "Mask"),
        os.path.join(train_dir, "Phys"))

    ZONES = ["充分", "部分阴影", "完全阴影"]
    # 累加器：n, Sμ, SX, Sμμ, SXX, SμX（μ 相关用）；SSE_mu / SSE_fin / SSE_base（RMSE 用）
    acc = {z: {"n": 0, "Smu": 0.0, "SX": 0.0, "Smm": 0.0, "SXX": 0.0, "SmX": 0.0,
               "SSE_mu": 0.0, "SSE_fin": 0.0, "SSE_base": 0.0} for z in ZONES}

    with torch.no_grad():
        for i, spec in enumerate(specs):
            sample = loader(spec)
            thetas = [{k: v.unsqueeze(0).to(args.device) for k, v in t.items()}
                      for t in sample["thetas"]]
            x_hat = sample["x_hat"].unsqueeze(0).to(args.device)
            X_clean = sample["X_clean"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(args.device)
            metal_mask = sample["metal_mask"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(args.device)

            Z = len(thetas)
            dirty_vol = x_hat.permute(0, 2, 1, 3, 4)                    # [1,1,Z,H,W]
            dirty = [dirty_vol[:, :, z] for z in range(Z)]              # [Z] 个 [1,1,H,W]
            masks = [metal_mask[:, :, z] for z in range(Z)]             # [Z] 个 [1,1,H,W]
            observer = trainer._observer(dirty, masks)
            res = forward_backward(m_net, thetas, d_ref,
                                   recursion=args.recursion, observer=observer)
            mu_fused = trainer._stack([s.mu for s in res.fused])
            d_j = trainer._stack([t["d_j"] for t in thetas])
            x_hat_final = trainer._final_output(res, dirty, masks)

            I = 1.0 - metal_mask
            r = d_j / d_ref
            mu = mu_fused
            X = X_clean
            xb = x_hat.permute(0, 2, 1, 3, 4)

            for name in ZONES:
                sel = I > 0.5
                if name == "充分":
                    sel = sel & (r > 9.0)
                elif name == "部分阴影":
                    sel = sel & (r >= 1.0 / 9.0) & (r <= 9.0)
                else:  # 完全阴影
                    sel = sel & (r < 1.0 / 9.0)

                mu_v = mu[sel].double()
                X_v = X[sel].double()
                fin_v = x_hat_final[sel].double()
                base_v = xb[sel].double()
                cnt = int(sel.sum().item())
                if cnt == 0:
                    continue
                a = acc[name]
                a["n"] += cnt
                a["Smu"] += float(mu_v.sum())
                a["SX"] += float(X_v.sum())
                a["Smm"] += float((mu_v * mu_v).sum())
                a["SXX"] += float((X_v * X_v).sum())
                a["SmX"] += float((mu_v * X_v).sum())
                a["SSE_mu"] += float(((mu_v - X_v) ** 2).sum())
                a["SSE_fin"] += float(((fin_v - X_v) ** 2).sum())
                a["SSE_base"] += float(((base_v - X_v) ** 2).sum())

            if (i + 1) % 25 == 0:
                print(f"  ... {i + 1}/{len(specs)} 序列", flush=True)

    print(f"\n[μ质量] {'分区':<8} {'体素数':>10} {'μ_RMSE':>10} {'μ_corr':>9} "
          f"{'μ_bias':>9} {'fin_RMSE':>10} {'base_RMSE':>11}", flush=True)
    results = {}
    for name in ZONES:
        a = acc[name]
        n = a["n"]
        if n == 0:
            results[name] = {"n_voxels": 0, "mu_rmse": None, "mu_corr": None,
                             "mu_bias": None, "fin_rmse": None, "base_rmse": None}
            print(f"{name:<8} {0:>10} {'—':>10} {'—':>9} {'—':>9} {'—':>10} {'—':>11}", flush=True)
            continue
        mu_rmse = float(np.sqrt(a["SSE_mu"] / n))
        fin_rmse = float(np.sqrt(a["SSE_fin"] / n))
        base_rmse = float(np.sqrt(a["SSE_base"] / n))
        mu_bias = (a["Smu"] - a["SX"]) / n
        cov = a["SmX"] - a["Smu"] * a["SX"] / n
        vmu = a["Smm"] - a["Smu"] ** 2 / n
        vX = a["SXX"] - a["SX"] ** 2 / n
        denom = float(np.sqrt(vmu * vX))
        mu_corr = float(cov / denom) if denom > 0 else float("nan")
        results[name] = {"n_voxels": n, "mu_rmse": mu_rmse, "mu_corr": mu_corr,
                         "mu_bias": mu_bias, "fin_rmse": fin_rmse, "base_rmse": base_rmse}
        print(f"{name:<8} {n:>10} {mu_rmse:>10.2f} {mu_corr:>9.4f} {mu_bias:>9.1f} "
              f"{fin_rmse:>10.2f} {base_rmse:>11.2f}", flush=True)

    if args.out_json:
        out = {"patient": args.patient, "trust_mode": args.trust_mode,
               "recursion": args.recursion, "use_velocity": use_velocity,
               "d_ref": d_ref, "results": results}
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[save] {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
