"""viz_fanbeam_compare.py — 同一 (Target, Mask) 切片：扇束(ASTRA) vs 平行束(skimage) 对比。

用现有 V2.2 数据的 Target/Mask，分别用两条管线重建 Baseline+LI，并排画出来。
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, "/root/autodl-tmp/PRISM/3D_Coil")
from batch_simulate import simulate_astra, write_raw          # 扇束
from parallel_beam import (simulate_sino, THETA, PIXEL_MM,    # 平行束
                           BASELINE_FILTER, LO, HI, mu_to_hu)
from skimage.transform import iradon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/root/autodl-tmp/PRISM/training_data_astra"
OUT = "/root/autodl-tmp/PRISM/v2.2"
PICKS = {"small": "sample_003090", "medium": "sample_003169",
         "large": "sample_003260", "giant": "sample_001308"}


def rd(sid, sub):
    p = os.path.join(D, "train", sub, f"{sid}.raw")
    if not os.path.exists(p):
        p = os.path.join(D, "val", sub, f"{sid}.raw")
    return np.fromfile(p, np.float32).reshape(512, 512)


def parallel(clean, mask, cat, seed):
    L = simulate_sino(clean, mask, cat, seed)
    mu = iradon(L / PIXEL_MM, theta=THETA, filter_name=BASELINE_FILTER, circle=True)
    b = np.clip(mu_to_hu(mu), LO, HI).astype(np.float32)
    b[clean <= -500.0] = -1000.0
    mb = (mask > 1e-3).astype(np.float32)
    li = b.copy()
    if mb.sum() > 0:
        from skimage.morphology import binary_dilation
        from skimage.transform import radon
        ms = radon(mb, theta=THETA, circle=True)
        tr = binary_dilation(ms > 1e-4, footprint=np.ones((3, 9), bool))
        ls = L.copy(); cols = np.arange(ls.shape[1])
        for v in range(ls.shape[0]):
            if not tr[v].any():
                continue
            keep = ~tr[v]
            if keep.sum() < 2:
                continue
            ls[v] = np.interp(cols, cols[keep], ls[v][keep])
        li = mu_to_hu(iradon(ls / PIXEL_MM, theta=THETA, filter_name="shepp-logan", circle=True))
        li = np.clip(li, LO, HI).astype(np.float32)
        li[clean <= -500.0] = -1000.0
        li[mb > 0] = b[mb > 0]
    return b, li


def fan(clean, mask, sid, work):
    os.makedirs(work, exist_ok=True)
    cp = os.path.join(work, "clean.raw"); mp = os.path.join(work, "mask.raw")
    write_raw(cp, clean.astype(np.float32)); write_raw(mp, mask.astype(np.float32))
    ap, lp, _ = simulate_astra(cp, mp, anatomy="h", work_dir=work,
                               generate_li=True, seed=int(sid.rsplit("_", 1)[1]))
    return np.fromfile(ap, np.float32).reshape(512, 512), np.fromfile(lp, np.float32).reshape(512, 512)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT, "viz_fan_vs_parallel.png"))
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "font.size": 9})
    sizes = list(PICKS)
    fig, axes = plt.subplots(len(sizes), 5, figsize=(17, 3.6 * len(sizes)))
    for i, size in enumerate(sizes):
        sid = PICKS[size]
        gt, bl_p, li_p, mk = (rd(sid, s) for s in ("Target", "Baseline", "LI", "Mask"))
        cat = json.load(open(os.path.join(D, "meta", f"{sid}.json"))).get("category", "medium")
        b_par, l_par = parallel(gt, mk, cat, int(sid.rsplit("_", 1)[1]))
        b_fan, l_fan = fan(gt, mk, sid, f"/root/autodl-tmp/PRISM/v2.2/_fan/{sid}")
        rows = [
            (gt, "GT", "gray", -200, 400),
            (b_fan, "Baseline — FAN (ASTRA)", "gray", -200, 400),
            (b_par, "Baseline — PARALLEL (skimage)", "gray", -200, 400),
            (np.abs(b_fan - gt), "|FAN − GT|", "inferno", 0, 400),
            (np.abs(b_par - gt), "|PARALLEL − GT|", "inferno", 0, 400),
        ]
        for j, (img, tt, cm, lo, hi) in enumerate(rows):
            ax = axes[i, j]
            ax.imshow(img, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(tt, fontsize=9)
            if j == 0:
                ax.set_ylabel(size, fontsize=12)
        print(f"[{size}] FAN err {np.abs(b_fan-gt)[mk<=1e-3].mean():.1f} HU | "
              f"PARALLEL err {np.abs(b_par-gt)[mk<=1e-3].mean():.1f} HU", flush=True)
    fig.suptitle("Fan-beam (ASTRA) vs Parallel-beam (skimage) on the SAME clean slice+mask", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("[save]", args.out)


if __name__ == "__main__":
    main()
