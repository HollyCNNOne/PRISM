#!/usr/bin/env python3
"""regen_skimage.py — regenerate Baseline + LI for training_data_astra with the
self-contained skimage parallel-beam pipeline (radon/iradon).

Scope: ANALYTIC pipeline ONLY. Reads train/Target + train/Mask and OVERWRITES
train/Baseline + train/LI. Does NOT touch Target/Mask/Phys or any catsim/xcist
function. Replaces the earlier fan-beam ``regen_li.py`` (DEPRECATED: fan-beam
geometry + 1000 views makes linear interpolation too accurate → no secondary
artifacts, even with the 'S-L' kernel).

Physics (single source of truth = parallel_beam.simulate_sino, the verified probe):
  split tissue/metal radon → beam hardening (tissue, β=1.5e-2) → I=I0·e^−L →
  Poisson + electronic noise (σ_e=20) → −log → photon-starvation mottle
  (σ_starve=3 in L>6) → size-dependent PSF blur (large 4.5 / giant 6.0) →
  iradon('hann') → HU, clip [−1000,3000], bg(clean≤−500)=−1000.

  LI: radon(mask>1e-3) → trace=dilate(3,9) → np.interp LINEAR across trace in the
  SAME L_noisy → iradon('shepp-logan' sharp) → HU → clip → bg → metal back-fill
  → REAL secondary artifacts (fake bright spots / broken stripes).

Deterministic per-sample seed = int(sample_id suffix), so nproc order does not
affect output. Server-only (agir env: skimage + scipy). Usage:

  /root/miniconda3/envs/agir/bin/python -u regen_skimage.py --nproc 16
  /root/miniconda3/envs/agir/bin/python -u regen_skimage.py --limit 2 --sizes large,giant
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
from skimage.transform import radon, iradon
from scipy.ndimage import binary_dilation

from parallel_beam import (simulate_sino, mu_to_hu, PIXEL_MM, THETA,
                           BASELINE_FILTER, LI_FILTER, LO, HI)

TRAIN = "/root/autodl-tmp/PRISM/training_data_astra/train"
META_DIR = "/root/autodl-tmp/PRISM/training_data_astra/meta"
DONE_DIR = "/root/autodl-tmp/PRISM/training_data_astra/.regen_skimage_done"


def read_raw(p, shape=(512, 512)):
    return np.fromfile(p, dtype=np.float32).reshape(shape)


def _regen_one(sample_id):
    done = os.path.join(DONE_DIR, f"{sample_id}.done")
    if os.path.exists(done):
        return sample_id, "skip"

    try:
        meta = json.load(open(os.path.join(META_DIR, f"{sample_id}.json")))
        cat = meta.get("category", "medium")
    except Exception:
        cat = "medium"

    clean = read_raw(os.path.join(TRAIN, "Target", f"{sample_id}.raw"))
    mask = read_raw(os.path.join(TRAIN, "Mask", f"{sample_id}.raw"))
    seed = int(sample_id.rsplit("_", 1)[1])

    try:
        # ---------- Baseline ----------
        L_noisy = simulate_sino(clean, mask, cat, seed)   # 测量 sinogram（含噪声/PSF）
        mu_recon = iradon(L_noisy / PIXEL_MM, theta=THETA,
                          filter_name=BASELINE_FILTER, circle=True)
        baseline = mu_to_hu(mu_recon)
        baseline = np.clip(baseline, LO, HI)
        baseline[clean <= -500.0] = -1000.0

        # ---------- LI (raw linear interp + sharp FBP) ----------
        metal_binary = (mask > 1e-3).astype(np.float32)
        if metal_binary.sum() == 0:
            li_hu = baseline.copy()          # no metal → LI ≡ artifact
        else:
            metal_sino = radon(metal_binary, theta=THETA, circle=True)
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
            li_mu = iradon(li_sino / PIXEL_MM, theta=THETA,
                           filter_name=LI_FILTER, circle=True)
            li_hu = mu_to_hu(li_mu)
            li_hu = np.clip(li_hu, LO, HI)
            li_hu[clean <= -500.0] = -1000.0
            li_hu[metal_binary > 0] = baseline[metal_binary > 0]   # back-fill

        baseline.astype(np.float32).tofile(
            os.path.join(TRAIN, "Baseline", f"{sample_id}.raw"))
        li_hu.astype(np.float32).tofile(
            os.path.join(TRAIN, "LI", f"{sample_id}.raw"))
    except Exception as e:
        return sample_id, f"fail: {type(e).__name__}: {e}"

    os.makedirs(DONE_DIR, exist_ok=True)
    open(done, "w").close()
    return sample_id, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N")
    ap.add_argument("--sizes", default="", help="csv filter on meta.category")
    args = ap.parse_args()

    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(TRAIN, "Mask"))
                 if f.endswith(".raw"))
    if args.sizes:
        sizes = set(s.strip() for s in args.sizes.split(","))
        ids = [s for s in ids if json.load(
            open(os.path.join(META_DIR, f"{s}.json"))).get("category") in sizes]
    if args.limit:
        ids = ids[:args.limit]

    os.makedirs(DONE_DIR, exist_ok=True)
    print(f"[regen_skimage] {len(ids)} samples, nproc={args.nproc}", flush=True)

    n_ok = n_skip = n_fail = 0
    with Pool(args.nproc) as p:
        for sid, st in p.imap_unordered(_regen_one, ids):
            if st == "ok":
                n_ok += 1
            elif st == "skip":
                n_skip += 1
            else:
                n_fail += 1
                print(f"[FAIL] {sid}: {st}", flush=True)
    print(f"[regen_skimage] done ok={n_ok} skip={n_skip} fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
