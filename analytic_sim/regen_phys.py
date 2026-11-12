#!/usr/bin/env python3
"""regen_phys.py — 用 skimage 平行束几何重生成 Stage 2 命根子 d_j / g。

Scope: ANALYTIC pipeline ONLY。读取 train/Target + train/Mask，用
``parallel_beam.simulate_sino`` 重现 Baseline 重建**实际吃到的那条测量 sinogram**
（同 seed、同 rng 顺序、同 PSF/σ_e/σ_starve/束硬化），在其上算数据充分度 d_j 与
门控 g。覆盖 train/Phys/{sample_id}_{d_j,g}.npy。不触碰 x_M0（mask·metal_mu 几何
无关，已正确）、tau_b/w_b（0 占位，径向 LFR 未派生）、以及任何 catsim/xcist 函数。

为什么必须重生成（prism-li-fanbeam-too-clean 的延续）：
  旧 d_j/g 是**扇束 ASTRA**（strip_fanflat, 1000 视角）几何下 AᵀW(A·1) 的产物，
  而新 Baseline/LI 已切到 **skimage 平行束**（720 视角）。d_j 的空间图案（尤其
  金属背后阴影的走向：扇束发散 vs 平行束不发散）随几何变，Stage 2 SSM 的观测精度
  d_j/d_ref 与阴影门 s=1{r≥r0} 必须与观测图同几何，否则「数据充分度」与「阴影」错位。

数学（scale-invariant，故平行束绝对量级无关紧要）：
  d_j = Aᵀ W (A·1) = iradon(w · radon(1), filter_name=None)   [未滤波反投 = 伴随 Aᵀ]
  w   = I0·exp(−L_noisy)                                      [纯 Poisson 权重，= V4 sigma_e=0]
  g   = 1 / (1 + (d_j/d_ref)^3),  d_ref = 0.01×median(d_j)    [agir.gate.gate_map, k=3]

d_ref 冻结（run_train.freeze_d_ref）在训练时从全部 d_j 场重新 median，故重生成后
必须删掉陈旧的 d_ref_ho_*.npy 缓存（本脚本自动删，见 main）。

Server-only（agir env: skimage + scipy）。用法：
  /root/miniconda3/envs/agir/bin/python -u regen_phys.py --nproc 16
  /root/miniconda3/envs/agir/bin/python -u regen_phys.py --limit 2 --sizes large,giant
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
from skimage.transform import radon, iradon

from parallel_beam import (simulate_sino, I0, THETA, N_VIEWS)

TRAIN = "/root/autodl-tmp/PRISM/training_data_astra/train"
META_DIR = "/root/autodl-tmp/PRISM/training_data_astra/meta"
DATA_DIR = "/root/autodl-tmp/PRISM/training_data_astra"
PHYS_DIR = os.path.join(TRAIN, "Phys")
DONE_DIR = os.path.join(DATA_DIR, ".regen_phys_done")

GATE_K = 3.0            # agir.gate 门控陡度（V4 定案）
D_REF_REL = 0.01        # auto_d_ref 相对中位数倍数


def read_raw(p, shape=(512, 512)):
    return np.fromfile(p, dtype=np.float32).reshape(shape)


def auto_d_ref(d_map):
    """= agir.gate.auto_d_ref：d_ref = 0.01 × median(d_map)（组织透明）。"""
    return float(D_REF_REL * np.percentile(d_map, 50.0))


def gate_map(d, d_ref, k=GATE_K):
    """= agir.gate.gate_map：g_j = 1/(1+(d_j/d_ref)^k)。"""
    r = np.asarray(d, dtype=np.float64) / d_ref
    rk = np.power(r, k)
    return (1.0 / (1.0 + rk)).astype(np.float32)


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
        # 与 Baseline 同一测量 sinogram（同 seed/rng 顺序）→ 权重 W 一致
        L_noisy = simulate_sino(clean, mask, cat, seed)

        # 纯 Poisson 权重（V4 sigma_e=0 → whiting 退化为 I0·exp(−y)）
        w = I0 * np.exp(-L_noisy)

        # A·1 = 前投单位像（圆内弦长）；Aᵀ = 未滤波反投（filter_name=None）
        ones = np.ones((512, 512), dtype=np.float32)
        proj_ones = radon(ones, theta=THETA, circle=True)
        d_j = iradon(w * proj_ones, theta=THETA, filter_name=None, circle=True)
        d_j = d_j.astype(np.float32)
        assert d_j.shape == (512, 512), f"d_j shape {d_j.shape}"

        d_ref = auto_d_ref(d_j)
        g = gate_map(d_j, d_ref, GATE_K)

        np.save(os.path.join(PHYS_DIR, f"{sample_id}_d_j.npy"),
                d_j[None, ...].astype(np.float32))
        np.save(os.path.join(PHYS_DIR, f"{sample_id}_g.npy"),
                g[None, ...].astype(np.float32))
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
    os.makedirs(PHYS_DIR, exist_ok=True)

    # 删陈旧冻结 d_ref 缓存，强制训练时从新 d_j 场重算 median
    for f in ("d_ref_ho_patients_head1.npy", "d_ref_ho_patients_head2.npy",
              "d_ref.npy"):
        p = os.path.join(DATA_DIR, f)
        if os.path.exists(p):
            os.remove(p)
            print(f"[d_ref] 删除陈旧缓存 {p}", flush=True)

    print(f"[regen_phys] {len(ids)} samples, nproc={args.nproc}", flush=True)
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
    print(f"[regen_phys] done ok={n_ok} skip={n_skip} fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
