#!/usr/bin/env python
"""eval_zprofile.py —— B 消融（H3）的 Z 轴误差连续性分析（断层判据）。

对每个植入体序列，算金属遮蔽区（d_j/d_ref < 9，非金属体素）逐片 RMSE，输出：

1. 聚合指标：
   - shadow_rmse：遮蔽区（部分+完全阴影）整体 RMSE(HU)。
   - mean_jump / max_jump：「断层」度量 = 相邻片误差跳变 |err_z − err_{z−1}| 的均值/最大
     （仅统计相邻两片都有遮蔽体素的跳变）。
2. 每序列逐片 profile（--out-npz 存 object 数组，供画图）。

「断层」是 B 消融的核心视觉判据：递归（Full PRISM / B1 / B2）把相邻片信息沿 Z 平滑传进
遮蔽区，逐片误差 profile 连续、jump 小；无递归（B3 per_slice）每片独立冷启动，遮蔽片
误差独立尖峰，相邻片误差跳变大 → 断层。B3 若在遮蔽区出现断层，即是最想要的视觉证据。

用法：
    python scripts/eval_zprofile.py \
        --checkpoint <ckpt> --data-dir ... --patient patients_head2 \
        --trust-mode kalman --recursion bidirectional \
        --out-json <json> --out-npz <npz>
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
    ap.add_argument("--out-npz", default=None)
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
    print(f"[zprofile] checkpoint={args.checkpoint}\n"
          f"  patient={args.patient!r} trust={args.trust_mode} recursion={args.recursion} "
          f"use_velocity={use_velocity} d_ref={d_ref:.6g}  {len(specs)} 序列", flush=True)

    loader = make_theta_sequence_loader(
        os.path.join(train_dir, "Target"), os.path.join(train_dir, "Baseline"),
        os.path.join(train_dir, "LI"), os.path.join(train_dir, "Mask"),
        os.path.join(train_dir, "Phys"))

    profiles = []          # list[list[float|nan]]：每序列逐片遮蔽区 RMSE
    jumps = []             # list[float]：相邻两片都有遮蔽体素的 |err_z − err_{z−1}|
    SSE = 0.0
    CNT = 0

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
            d_j = trainer._stack([t["d_j"] for t in thetas])
            x_hat_final = trainer._final_output(res, dirty, masks)

            I = 1.0 - metal_mask
            r = d_j / d_ref
            shadow = (I > 0.5) & (r < 9.0)          # 遮蔽区 = 部分 + 完全阴影
            err2 = (x_hat_final - X_clean) ** 2

            Z = err2.shape[2]
            prof = []
            for z in range(Z):
                sel = shadow[:, :, z, :, :]
                cnt = int(sel.sum().item())
                if cnt > 0:
                    e = float(err2[:, :, z, :, :][sel].sum().item())
                    rmse = float(np.sqrt(e / cnt))
                    prof.append(rmse)
                    SSE += e
                    CNT += cnt
                else:
                    prof.append(float("nan"))
            profiles.append(prof)
            for z in range(1, Z):
                if not np.isnan(prof[z]) and not np.isnan(prof[z - 1]):
                    jumps.append(abs(prof[z] - prof[z - 1]))

            if (i + 1) % 25 == 0:
                print(f"  ... {i + 1}/{len(specs)} 序列", flush=True)

    shadow_rmse = float(np.sqrt(SSE / CNT)) if CNT > 0 else float("nan")
    mean_jump = float(np.mean(jumps)) if jumps else float("nan")
    max_jump = float(np.max(jumps)) if jumps else float("nan")

    print(f"\n[zprofile] shadow_rmse={shadow_rmse:.3f} HU  "
          f"mean_jump={mean_jump:.3f}  max_jump={max_jump:.3f}  "
          f"n_jumps={len(jumps)}  n_seqs={len(specs)}", flush=True)

    result = {
        "patient": args.patient, "trust_mode": args.trust_mode,
        "recursion": args.recursion, "use_velocity": use_velocity, "d_ref": d_ref,
        "shadow_rmse_hu": shadow_rmse,
        "mean_jump_hu": mean_jump, "max_jump_hu": max_jump,
        "n_jumps": len(jumps), "n_seqs": len(specs),
    }
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[save] {args.out_json}", flush=True)
    if args.out_npz:
        np.savez(args.out_npz, profiles=np.array(profiles, dtype=object),
                 jumps=np.array(jumps, dtype=np.float64))
        print(f"[save] {args.out_npz}", flush=True)


if __name__ == "__main__":
    main()
