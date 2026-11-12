#!/usr/bin/env python3
"""render_analytic_mar.py — clean analytical MAR simulation (radon/iradon).

Scope: ANALYTIC pipeline ONLY. Reads Target/Mask and recomputes Baseline + LI.

Baseline physics (tuned for realism — dirty streaks, not lasers):
  1. HU -> mu  (metal mu VERY high, 7.02 mm^-1)
  2. split tissue/metal radon line integrals
  3. beam hardening (tissue only): L_t <- L_t·(1 - beta·L_t)  -> cupping
  4. I = I0·exp(-L)
  5. Poisson photon noise + ELECTRONIC noise floor sigma_e
     (sigma_e dominates the photon-starved metal shadow -> mottled dark band)
  6. L_noisy = -log(max(I_noisy,0.5)/I0)
  7. size-dependent sinogram PSF blur (large/giant strongly blurred)
  8. iradon('hann') -> mu -> HU, clip [-1000,3000], bg=-1000

LI: radon(mask)->trace, np.interp LINEAR across trace, iradon('shepp-logan' sharp)
    -> REAL secondary artifacts.

Grid: rows = chosen sizes, cols = Target | LI | New baseline.
"""
import os, sys, glob, json, argparse
import numpy as np

MU_WATER = 0.02
METAL_MU = 7.02          # mm^-1 = 350000 HU (Pt-W)
PIXEL_MM = 220.16 / 512

# Size-dependent detector PSF sigma (sinogram detector pixels).
PSF_BY_SIZE = {"small": 1.5, "medium": 2.0, "large": 4.5, "giant": 6.0}

SIGMA_E = 20.0           # electronic noise floor (counts) — mottles the starved shadow
BETA_BH = 1.5e-2         # beam-hardening coeff (tissue-only cupping); 0 = off
L_STARVE = 6.0           # line integral beyond which the detector is photon-starved
SIGMA_STARVE = 3.0       # direct L-domain noise injected into the starved shadow (mottle)


def hu_to_mu(hu):
    return MU_WATER * (1.0 + hu / 1000.0)


def mu_to_hu(mu):
    return (mu / MU_WATER - 1.0) * 1000.0


def read_raw(p, shape=(512, 512)):
    return np.fromfile(p, dtype=np.float32).reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--sizes", default="large,giant")
    ap.add_argument("--n-per-size", type=int, default=1)
    ap.add_argument("--I0", type=float, default=1e5)
    ap.add_argument("--n-views", type=int, default=720)
    ap.add_argument("--filter", default="hann")
    ap.add_argument("--li-filter", default="shepp-logan")
    ap.add_argument("--sigma-e", type=float, default=SIGMA_E)
    ap.add_argument("--beta-bh", type=float, default=BETA_BH)
    ap.add_argument("--sigma-starve", type=float, default=SIGMA_STARVE)
    ap.add_argument("--psf-sigma", type=float, default=None,
                    help="override PSF for ALL sizes")
    ap.add_argument("--out", default="/root/autodl-tmp/PRISM/streak_fix_v3")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    from skimage.transform import radon, iradon
    from scipy.ndimage import gaussian_filter1d, binary_dilation

    sizes = [s.strip() for s in args.sizes.split(",")]
    os.makedirs(args.out, exist_ok=True)

    meta_dir = os.path.join(args.data_dir, "meta")
    train = os.path.join(args.data_dir, "train")
    by_size = {s: [] for s in sizes}
    for f in glob.glob(os.path.join(meta_dir, "*.json")):
        m = json.load(open(f))
        if m.get("category") in by_size:
            by_size[m["category"]].append(m["sample_id"])

    picks = {}
    for s in sizes:
        best = []
        for sid in by_size[s]:
            mask = read_raw(os.path.join(train, "Mask", f"{sid}.raw"))
            best.append((float(mask.sum()), sid))
        best.sort(reverse=True)
        picks[s] = [sid for _, sid in best[:args.n_per_size]]
        print(f"[pick] {s}: {picks[s]}", flush=True)

    theta = np.linspace(0.0, 180.0, args.n_views, endpoint=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for s in sizes:
        sigma = args.psf_sigma if args.psf_sigma is not None else PSF_BY_SIZE.get(s, 2.0)
        for sid in picks[s]:
            clean = read_raw(os.path.join(train, "Target", f"{sid}.raw"))
            mask = read_raw(os.path.join(train, "Mask", f"{sid}.raw"))
            rng = np.random.RandomState(args.seed)

            # ---------- Baseline ----------
            tissue_mu = (1.0 - mask) * hu_to_mu(clean)
            metal_mu = mask * METAL_MU
            L_tissue = radon(tissue_mu, theta=theta, circle=True) * PIXEL_MM
            L_metal = radon(metal_mu, theta=theta, circle=True) * PIXEL_MM
            # beam hardening (tissue only): sublinear -> cupping
            if args.beta_bh > 0:
                L_tissue = L_tissue * (1.0 - args.beta_bh * L_tissue)
            L = L_tissue + L_metal

            I = args.I0 * np.exp(-L)
            I_noisy = rng.poisson(I)
            # electronic readout noise floor — dominates the photon-starved shadow
            I_noisy = I_noisy + rng.normal(0.0, args.sigma_e, I_noisy.shape)
            I_noisy = np.maximum(I_noisy, 0.5)
            L_noisy = -np.log(I_noisy / args.I0)
            # direct photon-starvation mottle: beyond the detector dynamic range
            # the shadow's line integral is itself noise — inject L-domain noise
            # into the starved region (blurred below into blotchy dark bands)
            if args.sigma_starve > 0:
                starved = L > L_STARVE
                L_noisy = L_noisy + rng.normal(0.0, args.sigma_starve, L.shape) * starved
            L_noisy = gaussian_filter1d(L_noisy, sigma=sigma, axis=1, mode="nearest")
            mu_recon = iradon(L_noisy / PIXEL_MM, theta=theta,
                              filter_name=args.filter, circle=True)
            hu_recon = mu_to_hu(mu_recon)
            hu_recon = np.clip(hu_recon, -1000.0, 3000.0)
            hu_recon[clean <= -500.0] = -1000.0

            # ---------- LI ----------
            metal_binary = (mask > 1e-3).astype(np.float32)
            metal_sino = radon(metal_binary, theta=theta, circle=True)
            trace = metal_sino > 1e-4
            trace = binary_dilation(trace, structure=np.ones((3, 9)), iterations=1)
            li_sino = L_noisy.copy()
            col_idx = np.arange(li_sino.shape[1])
            for v in range(li_sino.shape[0]):
                if not trace[v].any():
                    continue
                keep = ~trace[v]
                if keep.sum() < 2:
                    continue
                li_sino[v] = np.interp(col_idx, col_idx[keep], li_sino[v][keep])
            li_mu = iradon(li_sino / PIXEL_MM, theta=theta,
                           filter_name=args.li_filter, circle=True)
            li_hu = mu_to_hu(li_mu)
            li_hu = np.clip(li_hu, -1000.0, 3000.0)
            li_hu[clean <= -500.0] = -1000.0
            li_hu[metal_binary > 0] = hu_recon[metal_binary > 0]

            t = (clean > -100) & (mask <= 0.01)
            rmse_li = float(np.sqrt(((li_hu - clean)[t] ** 2).mean()))
            rmse_new = float(np.sqrt(((hu_recon - clean)[t] ** 2).mean()))
            print(f"[done] {s} {sid} psf={sigma} sigma_e={args.sigma_e} "
                  f"beta_bh={args.beta_bh} RMSE(nonmetal) li={rmse_li:.0f} "
                  f"new={rmse_new:.0f}", flush=True)

            rows.append((s, clean, li_hu, hu_recon))

    n = len(rows)
    lo, hi = -1000.0, 3000.0
    fig, axes = plt.subplots(n, 3, figsize=(13, 4.4 * n))
    if n == 1:
        axes = axes[None, :]
    for r, (s, clean, li, new) in enumerate(rows):
        for c, (img, t) in enumerate(zip([clean, li, new],
                                         ["Target", "LI", "New baseline"])):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(t, fontsize=13)
            if c == 0:
                ax.set_ylabel(s, fontsize=14, rotation=0, labelpad=28, va="center")
    fig.suptitle(f"baseline={args.filter} PSF={PSF_BY_SIZE} sigma_e={args.sigma_e} "
                 f"sigma_starve={args.sigma_starve} beta_bh={args.beta_bh} "
                 f"I0={args.I0:.0e}", fontsize=13)
    fig.tight_layout(rect=[0.05, 0, 1, 0.96])
    grid = os.path.join(args.out, "grid_4sizes.png")
    fig.savefig(grid, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[grid] {grid}", flush=True)
    print("[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
