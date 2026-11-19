#!/usr/bin/env python
"""viz_regions.py —— 在 CT 切片上可视化区域划分（几何环 vs 物理 d_j 区）。

图 1（几何，按距离）：金属核(红) / 回填 8px(橙) / 过渡区 8→18px(黄)。
图 2（物理，按 d_j）：完全阴影 r<1/9(蓝) / 部分阴影 1/9≤r≤9(绿) / 充分 r>9(不填)。

底图用 Baseline（带伪影），让「区域」和「问题」对得上。
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/root/autodl-tmp/prism_framework")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

D = "/root/autodl-tmp/PRISM/training_data_astra/train"
META = "/root/autodl-tmp/PRISM/training_data_astra/meta"
D_REF = 4.10023e+06


def rd(sid, sub):
    return np.fromfile(os.path.join(D, sub, f"{sid}.raw"), np.float32).reshape(512, 512)


def _sample_idx(sid):
    return int(sid.rsplit("_", 1)[1])


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--sid", default=None, help="指定 sample；默认自动选金属面积适中的")
    ap.add_argument("--out", default="/root/autodl-tmp/PRISM/training_data_astra/viz_v12/viz_regions.png")
    args = ap.parse_args()

    # 自动选一个「金属适中、过渡区/完全阴影/充分区都非空」的样本
    if args.sid is None:
        metas = [json.load(open(f)) for f in glob.glob(os.path.join(META, "*.json"))]
        chosen = None
        for m in metas:
            if m.get("split") != "train":
                continue
            s = m["sample_id"]
            msk = rd(s, "Mask")
            a = (msk > 0.5).sum()
            if not (80 <= a <= 30000):
                continue
            dj = np.load(os.path.join(D, "Phys", f"{s}_d_j.npy")).reshape(512, 512)
            r = dj / D_REF
            nonmetal = msk < 0.5
            if (nonmetal & (r < 1.0 / 9.0)).sum() < 300:
                continue
            chosen = (a, s)
            break
        assert chosen is not None, "没找到合适的样本"
        sid = chosen[1]
        print(f"[viz] 自动选样本 {sid}，金属面积={chosen[0]}", flush=True)
    else:
        sid = args.sid
        print(f"[viz] 指定样本 {sid}", flush=True)

    base = rd(sid, "Baseline")
    msk = rd(sid, "Mask")
    dj = np.load(os.path.join(D, "Phys", f"{sid}_d_j.npy")).reshape(512, 512)
    r = dj / D_REF

    metal = msk > 0.5
    backfill = binary_dilation(metal, iterations=8)
    ring18 = binary_dilation(metal, iterations=18)
    geometric_ring = (ring18 & ~backfill)                # 纯几何环 8→18px（不限制光子）
    transition = geometric_ring & (r < 1.0 / 9.0)       # 过渡区 = 几何环 ∩ 完全阴影

    nonmetal = ~metal
    complete = nonmetal & (r < 1.0 / 9.0)
    partial = nonmetal & (r >= 1.0 / 9.0) & (r <= 9.0)
    sufficient = nonmetal & (r > 9.0)

    print(f"[viz] 金属={metal.sum()} 回填={backfill.sum()} 过渡区={transition.sum()} "
          f"完全阴影={complete.sum()} 部分阴影={partial.sum()} 充分={sufficient.sum()}", flush=True)

    def overlay(ax, img, fills):
        ax.imshow(img, cmap="gray", vmin=-150, vmax=350, interpolation="nearest")
        for mask, color, alpha in fills:
            ax.imshow(np.where(mask, 1.0, np.nan), cmap=__import__("matplotlib.colors", fromlist=["ListedColormap"])
                      .ListedColormap([color]), alpha=alpha, interpolation="nearest")

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    # 放大到金属周围（金属很小，全图看不清环）
    ys, xs = np.nonzero(metal)
    cy, cx = int(ys.mean()), int(xs.mean())
    R = 90
    for ax in axes:
        ax.set_xlim(cx - R, cx + R)
        ax.set_ylim(cy + R, cy - R)

    # 图 1：几何环
    ax = axes[0]
    ax.imshow(base, cmap="gray", vmin=-150, vmax=350, interpolation="nearest")
    ax.imshow(np.where(metal, 1.0, np.nan), cmap=ListedColormap(["red"]), alpha=0.9, interpolation="nearest")
    ax.imshow(np.where(backfill & ~metal, 1.0, np.nan), cmap=ListedColormap(["orange"]), alpha=0.5, interpolation="nearest")
    ax.imshow(np.where(geometric_ring, 1.0, np.nan), cmap=ListedColormap(["yellow"]), alpha=0.6, interpolation="nearest")
    ax.set_title("Geometric rings (by distance)\nred=metal core  orange=backfill 8px  yellow=ring 8-18px",
                 fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # 图 2：物理 d_j 区
    ax = axes[1]
    ax.imshow(base, cmap="gray", vmin=-150, vmax=350, interpolation="nearest")
    ax.imshow(np.where(metal, 1.0, np.nan), cmap=ListedColormap(["red"]), alpha=0.9, interpolation="nearest")
    ax.imshow(np.where(complete, 1.0, np.nan), cmap=ListedColormap(["blue"]), alpha=0.5, interpolation="nearest")
    ax.imshow(np.where(partial, 1.0, np.nan), cmap=ListedColormap(["green"]), alpha=0.4, interpolation="nearest")
    ax.set_title("Physical d_j zones (by photon count)\nred=metal  blue=complete-shadow r<1/9  green=partial-shadow 1/9<=r<=9",
                 fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"{sid} — region visualization (base = Baseline artifact)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {args.out}", flush=True)


if __name__ == "__main__":
    from matplotlib.colors import ListedColormap
    main()
