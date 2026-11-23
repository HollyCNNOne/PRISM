"""viz_streak_sweep.py — 平行束条纹密度参数扫描（视角数 × 角度向 PSF）。

同一张 small 切片，网格里换 N_VIEWS 与 PSF σ，看哪种条纹密度像目标。
"""
import argparse, os, sys
import numpy as np
from skimage.transform import radon, iradon
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, "/root/autodl-tmp/PRISM/3D_Coil")
from parallel_beam import (hu_to_mu, mu_to_hu, METAL_MU, PIXEL_MM, I0,
                           SIGMA_E, BETA_BH, L_STARVE, SIGMA_STARVE,
                           BASELINE_FILTER, LO, HI)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/root/autodl-tmp/PRISM/training_data_astra"
OUT = "/root/autodl-tmp/PRISM/v2.2"


def rd(sid, sub):
    p = os.path.join(D, "train", sub, f"{sid}.raw")
    if not os.path.exists(p):
        p = os.path.join(D, "val", sub, f"{sid}.raw")
    return np.fromfile(p, np.float32).reshape(512, 512)


def render(clean, mask, seed, n_views=720, psf=1.5, psf_axis=1):
    rng = np.random.RandomState(seed)
    theta = np.linspace(0.0, 180.0, n_views, endpoint=False)
    tissue = (1.0 - mask) * hu_to_mu(clean)
    metal = mask * METAL_MU
    Lt = radon(tissue, theta=theta, circle=True) * PIXEL_MM
    Lm = radon(metal, theta=theta, circle=True) * PIXEL_MM
    if BETA_BH > 0:
        Lt = Lt * (1.0 - BETA_BH * Lt)
    L = Lt + Lm
    I = I0 * np.exp(-L)
    In = rng.poisson(I)
    In = In + rng.normal(0.0, SIGMA_E, In.shape)
    In = np.maximum(In, 0.5)
    Ln = -np.log(In / I0)
    if SIGMA_STARVE > 0:
        Ln = Ln + rng.normal(0.0, SIGMA_STARVE, L.shape) * (L > L_STARVE)
    Ln = gaussian_filter1d(Ln, sigma=psf, axis=psf_axis, mode="nearest")
    mu = iradon(Ln / PIXEL_MM, theta=theta, filter_name=BASELINE_FILTER, circle=True)
    b = np.clip(mu_to_hu(mu), LO, HI).astype(np.float32)
    b[clean <= -500.0] = -1000.0
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", default="sample_003090", help="small")
    ap.add_argument("--out", default=os.path.join(OUT, "viz_streak_sweep.png"))
    args = ap.parse_args()
    gt = rd(args.sid, "Target"); mk = rd(args.sid, "Mask")
    seed = int(args.sid.rsplit("_", 1)[1])
    views = [720, 360, 180]
    psfs = [1.5, 3.0, 6.0]

    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "font.size": 9})
    fig, axes = plt.subplots(len(psfs), len(views) + 1, figsize=(3.4 * (len(views) + 1), 3.4 * len(psfs)))
    for i, psf in enumerate(psfs):
        for j, v in enumerate(views):
            b = render(gt, mk, seed, n_views=v, psf=psf)
            ax = axes[i, j]
            ax.imshow(b, cmap="gray", vmin=-200, vmax=400, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            err = np.abs(b - gt)[mk <= 1e-3].mean()
            ax.set_title(f"{v} views, PSF σ={psf}\nerr {err:.0f} HU", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"PSF σ={psf}", fontsize=10)
        ax = axes[i, -1]
        ax.imshow(np.abs(render(gt, mk, seed, n_views=720, psf=psf) - gt),
                  cmap="inferno", vmin=0, vmax=300, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("|err| @720", fontsize=8)
    fig.suptitle(f"{args.sid} — streak density sweep (parallel beam)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("[save]", args.out)


if __name__ == "__main__":
    main()
