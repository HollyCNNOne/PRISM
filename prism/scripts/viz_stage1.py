#!/usr/bin/env python
"""viz_stage1.py —— Stage 1 多尺寸视觉验收（定性）。

认知边界：Stage 1 是纯 2D FIND-Net（无 Z 轴、无物理常识），只验收「金属外围放射状条纹
有没有被擦掉」；不碰牙齿/骨骼区、不评 HU 精度。

对 4 种线圈尺寸（small/medium/large/giant）各从验证集（--val-patient，默认 patients_head2）
挑一张「条纹最重」的切片 —— 判据 = 非金属区 (Baseline−Target)² 均值（越大条纹越重），
然后输出 3 列对比图：Baseline(污染输入) | Prediction(FIND-Net) | Ground Truth(真值)。
拼成一张 4 行×3 列大图，另存每尺寸单图。

用法：
    python scripts/viz_stage1.py --checkpoint .../findnet_stage1.pt --device cuda
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prism.models.find_net import FINDNet, FINDNetArgs  # noqa: E402

SIZES = ["small", "medium", "large", "giant"]


def _read_np(path, h=512, w=512):
    return np.fromfile(path, dtype=np.float32).reshape(h, w)


def load_meta(meta_dir, val_patient):
    metas = {}
    for f in glob.glob(os.path.join(meta_dir, "*.json")):
        m = json.load(open(f))
        if m.get("patient_id") == val_patient:
            metas[m["sample_id"]] = m
    return metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--val-patient", default="patients_head2")
    ap.add_argument("--out", default=None, help="拼图输出路径（默认 data_dir/viz/stage1_4sizes.png）")
    ap.add_argument("--window-min", type=float, default=-150.0)
    ap.add_argument("--window-max", type=float, default=350.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    find_net = FINDNet(FINDNetArgs()).to(args.device)
    find_net.load_state_dict(ck["find_net"])
    find_net.eval()
    ep = ck.get("config", {}).get("epoch", "?")
    print(f"[load] {args.checkpoint} (epoch {ep})", flush=True)

    train_dir = os.path.join(args.data_dir, "train")
    metas = load_meta(os.path.join(args.data_dir, "meta"), args.val_patient)

    # 1. 每尺寸挑「条纹最重」切片（非金属区 Baseline−Target 方差最大）
    picks = {}
    for size in SIZES:
        sids = [sid for sid, m in metas.items() if m.get("category") == size]
        best_sid, best_e = None, -1.0
        for sid in sids:
            base = _read_np(os.path.join(train_dir, "Baseline", f"{sid}.raw"))
            tgt = _read_np(os.path.join(train_dir, "Target", f"{sid}.raw"))
            msk = _read_np(os.path.join(train_dir, "Mask", f"{sid}.raw"))
            w = 1.0 - msk
            e = float(((base - tgt) ** 2 * w).sum() / (w.sum() + 1e-9))
            if e > best_e:
                best_e, best_sid = e, sid
        picks[size] = (best_sid, best_e)
        print(f"[pick] {size:<6} {best_sid}  streak_energy={best_e:.1f}  (n={len(sids)})", flush=True)

    # 2. 出图
    out = args.out or os.path.join(args.data_dir, "viz", "stage1_4sizes.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig, axes = plt.subplots(len(SIZES), 3, figsize=(13, 4.2 * len(SIZES) + 1.2))

    cols = ["Baseline (input)", "Prediction (FIND-Net)", "Ground Truth"]
    for c, title in enumerate(cols):
        axes[0, c].set_title(title, fontsize=13, weight="bold")

    for r, size in enumerate(SIZES):
        sid, e = picks[size]
        base = _read_np(os.path.join(train_dir, "Baseline", f"{sid}.raw"))
        lict = _read_np(os.path.join(train_dir, "LI", f"{sid}.raw"))
        msk = _read_np(os.path.join(train_dir, "Mask", f"{sid}.raw"))
        tgt = _read_np(os.path.join(train_dir, "Target", f"{sid}.raw"))

        with torch.no_grad():
            ct_ma = torch.from_numpy(base).float()[None, None].to(args.device)
            lict_t = torch.from_numpy(lict).float()[None, None].to(args.device)
            msk_t = torch.from_numpy(msk).float()[None, None].to(args.device)
            pred = find_net.reconstruct(ct_ma, lict_t, msk_t).squeeze().cpu().numpy()

        imgs = [base, pred, tgt]
        for c, arr in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(arr, cmap="gray", vmin=args.window_min, vmax=args.window_max,
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"{size}\n{sid}\nstreak={e:.0f}", fontsize=11)

    plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.12, right=0.98, top=0.94, bottom=0.02)
    fig.suptitle(f"Stage 1 FIND-Net (epoch {ep}) - 4-size streak check - window [{args.window_min}, {args.window_max}] HU",
                 fontsize=13, weight="bold")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"[save] {out}", flush=True)

    # 每尺寸单图
    single_dir = os.path.join(os.path.dirname(out), "per_size")
    os.makedirs(single_dir, exist_ok=True)
    for r, size in enumerate(SIZES):
        sid, e = picks[size]
        base = _read_np(os.path.join(train_dir, "Baseline", f"{sid}.raw"))
        lict = _read_np(os.path.join(train_dir, "LI", f"{sid}.raw"))
        msk = _read_np(os.path.join(train_dir, "Mask", f"{sid}.raw"))
        tgt = _read_np(os.path.join(train_dir, "Target", f"{sid}.raw"))
        with torch.no_grad():
            ct_ma = torch.from_numpy(base).float()[None, None].to(args.device)
            lict_t = torch.from_numpy(lict).float()[None, None].to(args.device)
            msk_t = torch.from_numpy(msk).float()[None, None].to(args.device)
            pred = find_net.reconstruct(ct_ma, lict_t, msk_t).squeeze().cpu().numpy()
        f2, ax2 = plt.subplots(1, 3, figsize=(12, 4.4))
        for c, (title, arr) in enumerate(zip(cols, [base, pred, tgt])):
            ax2[c].imshow(arr, cmap="gray", vmin=args.window_min, vmax=args.window_max,
                          interpolation="nearest")
            ax2[c].set_title(title, fontsize=12)
            ax2[c].set_xticks([])
            ax2[c].set_yticks([])
        f2.suptitle(f"{size} — {sid}", fontsize=13, weight="bold")
        p = os.path.join(single_dir, f"stage1_{size}.png")
        f2.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(f2)
        print(f"[save] {p}", flush=True)

    plt.close(fig)


if __name__ == "__main__":
    main()
