#!/usr/bin/env python3
"""render_z_continuity.py — 展示解析端（skimage 平行束 V4）的 z 轴连续。

同一植入体（sequence_id）的连续 z 切片，按 z 递增横向排开，3 行 = Target | Baseline | LI。
Target 行看线圈/组织在 z 上的连续演化；Baseline/LI 行看解析伪影在 z 上的连续（星芒方向、
暗带位置随切片平滑变化，而非逐片跳变）——这是 PRISM SSM 递归先验利用的连续性。

Scope: ANALYTIC pipeline ONLY。V4 物理走 parallel_beam.simulate_sino（与 regen_skimage/
regen_phys 完全同 seed、同噪声）。

Server-only（agir env）。用法：
  /root/miniconda3/envs/agir/bin/python -u render_z_continuity.py --sequence-id 54
"""
import argparse
import json
import glob
import os

import numpy as np
from skimage.transform import radon, iradon
from scipy.ndimage import binary_dilation

from parallel_beam import (simulate_sino, mu_to_hu, PIXEL_MM, THETA,
                           BASELINE_FILTER, LI_FILTER, LO, HI)

DATA = "/root/autodl-tmp/PRISM/training_data_astra"
TRAIN = os.path.join(DATA, "train")
META_DIR = os.path.join(DATA, "meta")


def read_raw(p, shape=(512, 512)):
    return np.fromfile(p, dtype=np.float32).reshape(shape)


def render_one(sample_id, cat, seed):
    """返回 (clean, baseline, li) 三张 HU 图，V4 物理。"""
    clean = read_raw(os.path.join(TRAIN, "Target", f"{sample_id}.raw"))
    mask = read_raw(os.path.join(TRAIN, "Mask", f"{sample_id}.raw"))

    L_noisy = simulate_sino(clean, mask, cat, seed)
    mu_recon = iradon(L_noisy / PIXEL_MM, theta=THETA,
                      filter_name=BASELINE_FILTER, circle=True)
    baseline = np.clip(mu_to_hu(mu_recon), LO, HI)
    baseline[clean <= -500.0] = -1000.0

    metal_binary = (mask > 1e-3).astype(np.float32)
    if metal_binary.sum() == 0:
        li = baseline.copy()
    else:
        metal_sino = radon(metal_binary, theta=THETA, circle=True)
        trace = binary_dilation(metal_sino > 1e-4, structure=np.ones((3, 9)), iterations=1)
        li_sino = L_noisy.copy()
        col_idx = np.arange(li_sino.shape[1])
        for v in range(li_sino.shape[0]):
            if not trace[v].any():
                continue
            keep = ~trace[v]
            if keep.sum() < 2:
                continue
            li_sino[v] = np.interp(col_idx, col_idx[keep], li_sino[v][keep])
        li_mu = iradon(li_sino / PIXEL_MM, theta=THETA,
                       filter_name=LI_FILTER, circle=True)
        li = np.clip(mu_to_hu(li_mu), LO, HI)
        li[clean <= -500.0] = -1000.0
        li[metal_binary > 0] = baseline[metal_binary > 0]
    return clean, baseline, li


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence-id", type=int, default=54)
    ap.add_argument("--stride", type=int, default=1, help="z 下采样步长")
    ap.add_argument("--out", default="/root/autodl-tmp/PRISM/z_continuity")
    args = ap.parse_args()

    metas = []
    for f in glob.glob(os.path.join(META_DIR, "*.json")):
        m = json.load(open(f))
        if m.get("sequence_id") == args.sequence_id:
            metas.append(m)
    metas.sort(key=lambda m: m["sample_id"])   # sample_id 升序 = z 递增
    ids = [m["sample_id"] for m in metas][:: args.stride]

    cat = metas[0].get("category", "medium")
    patient = metas[0].get("patient_id", "?")
    print(f"[z-cont] seq={args.sequence_id} cat={cat} patient={patient} "
          f"n={len(metas)} shown={len(ids)}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [[], [], []]   # Target / Baseline / LI
    for sid in ids:
        seed = int(sid.rsplit("_", 1)[1])
        clean, base, li = render_one(sid, cat, seed)
        rows[0].append(clean)
        rows[1].append(base)
        rows[2].append(li)

    names = ["Target", "Baseline", "LI"]
    n = len(ids)
    lo, hi = -1000.0, 3000.0
    fig, axes = plt.subplots(3, n, figsize=(1.9 * n, 6.4))
    for r in range(3):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(rows[r][c], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"z={c}", fontsize=11)
            if c == 0:
                ax.set_ylabel(names[r], fontsize=13, rotation=0, labelpad=24, va="center")
    fig.suptitle(f"sequence {args.sequence_id} [{cat}] {patient} — z-axis continuity "
                 f"(stride={args.stride}, V4 physics)", fontsize=13)
    fig.tight_layout(rect=[0.04, 0, 1, 0.95])
    os.makedirs(args.out, exist_ok=True)
    png = os.path.join(args.out, f"z_cont_seq{args.sequence_id}.png")
    fig.savefig(png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[png] {png}", flush=True)
    print("[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
