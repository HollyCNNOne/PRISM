"""viz_fanbeam.py — 用扇束(ASTRA DD2FanProj + fdk_equiAngle)重建，出图看伪影形态。"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, "/root/autodl-tmp/PRISM/3D_Coil")
from batch_simulate import simulate_astra, write_raw

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT, "viz_fanbeam.png"))
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "font.size": 9})
    sizes = list(PICKS)
    fig, axes = plt.subplots(len(sizes), 5, figsize=(17, 3.5 * len(sizes)))
    for i, size in enumerate(sizes):
        sid = PICKS[size]
        gt = rd(sid, "Target"); mk = rd(sid, "Mask")
        work = f"/root/autodl-tmp/PRISM/v2.2/_fan/{sid}"
        os.makedirs(work, exist_ok=True)
        cp = os.path.join(work, "clean.raw"); mp = os.path.join(work, "mask.raw")
        write_raw(cp, gt.astype(np.float32)); write_raw(mp, mk.astype(np.float32))
        ap_, lp_, _ = simulate_astra(cp, mp, anatomy="h", work_dir=work,
                                     generate_li=True, seed=int(sid.rsplit("_", 1)[1]))
        bl = np.fromfile(ap_, np.float32).reshape(512, 512)
        li = np.fromfile(lp_, np.float32).reshape(512, 512)
        tis = mk <= 1e-3
        print(f"[{size}] FAN baseline err {np.abs(bl-gt)[tis].mean():.1f} HU  "
              f"metal_sum={mk.sum():.1f}", flush=True)
        rows = [
            (gt, "GT", "gray", -200, 400),
            (bl, "Baseline — FAN (ASTRA)", "gray", -200, 400),
            (li, "LI — FAN", "gray", -200, 400),
            (np.abs(bl - gt), "|Baseline − GT|", "inferno", 0, 400),
            (mk, "Mask", "viridis", 0, max(mk.max(), 1e-3)),
        ]
        for j, (img, tt, cm, lo, hi) in enumerate(rows):
            ax = axes[i, j]
            ax.imshow(img, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(tt, fontsize=9)
            if j == 0:
                ax.set_ylabel(size, fontsize=12)
    fig.suptitle("Fan-beam (ASTRA) reconstruction — V2.2 targets/masks", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("[save]", args.out)


if __name__ == "__main__":
    main()
