#!/usr/bin/env python
"""v13_eval_ruler_viz.py —— 评估尺子可视化（Mask Overlay）。

对 sample_000973 出图：Baseline 灰底 + 半透明覆盖
  - metal_core（蓝，α=0.6）：mask > 0.5，评估必须避开的金属高光区
  - eval_ring（红，α=0.6）：dilate(metal, r_backfill+ring) − dilate(metal, r_backfill) ∩ r<1/9，
    修正后的过渡区尺子（只落在网络恢复的脑组织上，避开回填晕影）
全图 + 金属边缘 Zoom-in 两面板，用于亲眼核对评估区域避开了高光、精准落在组织上。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/autodl-tmp/prism_framework")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rd(path, h=512, w=512):
    return np.fromfile(path, dtype=np.float32).reshape(h, w)


def dilate(binm, r):
    t = torch.from_numpy(binm.astype(np.float32))[None, None]
    k = 2 * r + 1
    return F.max_pool2d(t, kernel_size=k, stride=1, padding=r)[0, 0].numpy() > 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="/root/autodl-tmp/PRISM/training_data_astra/ablations_v13/00_baseline.pt")
    ap.add_argument("--sid", default="sample_000973")
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--ring-width", type=int, default=10)
    ap.add_argument("--out",
                    default="/root/autodl-tmp/PRISM/training_data_astra/viz_v12/v13_eval_ruler.png")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    backfill_r = ckpt["config"].get("backfill_r", 8)
    d_ref = ckpt["config"]["d_ref"]

    train_dir = os.path.join(args.data_dir, "train")
    base = rd(os.path.join(train_dir, "Baseline", f"{args.sid}.raw"))
    msk = rd(os.path.join(train_dir, "Mask", f"{args.sid}.raw"))
    d_j = np.load(os.path.join(train_dir, "Phys", f"{args.sid}_d_j.npy")).reshape(512, 512)

    core = msk > 0.5
    r = d_j / d_ref
    ring = (dilate(core, backfill_r + args.ring_width) & ~dilate(core, backfill_r)) & (r < 1.0 / 9.0)

    print(f"[ruler] backfill_r={backfill_r} ring_width={args.ring_width}", flush=True)
    print(f"[ruler] metal_core 像素={int(core.sum())}  eval_ring 像素={int(ring.sum())}  "
          f"重叠={int((core & ring).sum())}（应=0，评估区完全避开高光）", flush=True)
    ys, xs = np.nonzero(core)
    cy, cx = int(ys.mean()), int(xs.mean())

    # RGBA overlay
    H, W = base.shape
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    overlay[core] = [0.0, 0.0, 1.0, 0.6]   # blue = metal_core
    overlay[ring] = [1.0, 0.0, 0.0, 0.6]   # red = eval_ring

    fig, axes = plt.subplots(1, 2, figsize=(22, 11))
    for ax in axes:
        ax.imshow(base, cmap="gray", vmin=-150, vmax=350, interpolation="nearest")
        ax.imshow(overlay, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_title("full (blue=metal_core, red=eval_ring)", fontsize=12)
    axes[1].set_title("zoom metal-edge", fontsize=12)
    r_zoom = 90
    axes[1].set_xlim(max(0, cx - r_zoom), min(W, cx + r_zoom))
    axes[1].set_ylim(max(0, cy - r_zoom), min(H, cy + r_zoom))
    fig.suptitle(f"v1.3 eval ruler | {args.sid} | backfill_r={backfill_r} "
                 f"ring=[{backfill_r},{backfill_r + args.ring_width}]px | "
                 f"overlap={int((core & ring).sum())} (must be 0)", fontsize=13)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {args.out}", flush=True)


if __name__ == "__main__":
    main()
