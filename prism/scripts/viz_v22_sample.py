"""viz_v22_sample.py — 出一张 V2.2 仿真样本图（GT / Baseline / LI / 差值 / Mask）。"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/root/autodl-tmp/PRISM/training_data_astra"


def rd(sid, sub):
    return np.fromfile(os.path.join(D, "train", sub, f"{sid}.raw"), np.float32).reshape(512, 512)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", default="sample_003260")
    ap.add_argument("--out", default="/root/autodl-tmp/PRISM/v2.2/sample_viz.png")
    args = ap.parse_args()
    sid = args.sid

    gt = rd(sid, "Target"); bl = rd(sid, "Baseline")
    li = rd(sid, "LI"); mk = rd(sid, "Mask")

    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "font.size": 9})
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.0))
    panels = [
        (gt, "GT (clean)", "gray", -200, 400),
        (bl, "Baseline (artifact)", "gray", -200, 400),
        (li, "LI (classic MAR)", "gray", -200, 400),
        (np.abs(bl - gt), "|Baseline − GT|", "inferno", 0, 400),
        (mk, "Mask (metal fraction)", "viridis", 0, max(mk.max(), 1e-3)),
    ]
    for ax, (img, title, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(f"V2.2 analytic sample {sid} — parallel-beam (720 views), "
                 f"thickness-resampled, PSF-blurred", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[save]", args.out)


if __name__ == "__main__":
    main()
