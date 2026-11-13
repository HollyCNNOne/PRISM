#!/usr/bin/env python
"""apply_bias_head.py —— 把训好的 bias 校正头当固定预处理，对 held-out 观测做
x̂_corr = x̂ − δ_pred·mask(部分阴影)。

OOD 判据（便宜端到端检查）：生成 Baseline_bcorr/ 后，拿 B8 现有 ckpt 跑 eval_mar.py /
eval_mu_quality.py（--baseline-dir 指向 Baseline_bcorr），看完全阴影 μ_bias 是否从 +77
移动、fin_RMSE 是否从 616 移动。

为什么只改部分阴影、只读 x_M0/d_j：与 bias_head_probe.py 训练口径一致（δ 只在部分阴影
被监督，完全阴影/充分区/金属不监督 → δ_pred 只在部分阴影可信，必须 mask）。只读几何
→ 只能输出「金属几何决定的那部分系统偏置」，不会学成恒等。

用法（服务器 pzneuro env）：
    python apply_bias_head.py --data-dir /root/autodl-tmp/PRISM/training_data \
        --ckpt /root/autodl-tmp/PRISM/training_data/bias_head_probe.pt \
        --patient patients_head2 --device cuda
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bias_head_probe import BiasHead  # noqa: E402

H = W = 512


def list_heldout(meta_dir: str, patient: str):
    ids = []
    for f in glob.glob(os.path.join(meta_dir, "*.json")):
        m = json.load(open(f))
        if m.get("split", "train") != "train":
            continue
        if m.get("patient_id") == patient:
            ids.append(m["sample_id"])
    ids.sort()
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data")
    ap.add_argument("--ckpt", default="/root/autodl-tmp/PRISM/training_data/bias_head_probe.pt")
    ap.add_argument("--patient", default="patients_head2")
    ap.add_argument("--out-subdir", default="Baseline_bcorr")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    d_ref = float(ckpt["d_ref"])
    base = int(ckpt["base"])
    net = BiasHead(in_ch=2, base=base).to(args.device)
    net.load_state_dict(ckpt["model"])
    net.eval()

    train_dir = os.path.join(args.data_dir, "train")
    meta_dir = os.path.join(args.data_dir, "meta")
    baseline_dir = os.path.join(train_dir, "Baseline")
    phys_dir = os.path.join(train_dir, "Phys")
    out_dir = os.path.join(train_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    ids = list_heldout(meta_dir, args.patient)
    print(f"[apply] patient={args.patient} n={len(ids)} d_ref={d_ref:.6g}", flush=True)

    n_partial = 0
    sum_abs = 0.0
    sum_abs_hat = 0.0
    with torch.no_grad():
        for i, sid in enumerate(ids):
            x_hat = np.fromfile(os.path.join(baseline_dir, f"{sid}.raw"),
                                dtype=np.float32).reshape(H, W)
            x_M0 = np.load(os.path.join(phys_dir, f"{sid}_x_M0.npy")).astype(np.float32)[0]
            d_j = np.load(os.path.join(phys_dir, f"{sid}_d_j.npy")).astype(np.float32)[0]

            r = d_j / d_ref
            metal = x_M0 > 0
            partial = (r >= 1.0 / 9.0) & (r < 9.0) & (~metal)

            ch0 = (x_M0 / 35000.0).astype(np.float32)
            ch1 = (np.log1p(r) / 5.0).astype(np.float32)
            inputs = torch.from_numpy(np.stack([ch0, ch1], axis=0)).unsqueeze(0).to(args.device)

            pred = net(inputs)[0, 0].cpu().numpy()          # [H,W] δ_pred
            delta_corr = pred * partial.astype(np.float32)  # 只在部分阴影生效
            x_corr = x_hat - delta_corr
            x_corr.astype(np.float32).tofile(os.path.join(out_dir, f"{sid}.raw"))

            n_partial += int(partial.sum())
            sum_abs += float(np.abs(pred[partial]).sum())
            sum_abs_hat += float(np.abs(x_hat[partial]).sum())
            if (i + 1) % 200 == 0:
                print(f"  ... {i + 1}/{len(ids)}", flush=True)

    print(f"[done] wrote {len(ids)} slices -> {out_dir}", flush=True)
    print(f"[stat] partial voxels={n_partial}  mean|δ_pred|={sum_abs/max(n_partial,1):.2f} HU  "
          f"(mean|x̂|={sum_abs_hat/max(n_partial,1):.2f} HU)", flush=True)


if __name__ == "__main__":
    main()
