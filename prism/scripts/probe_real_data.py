#!/usr/bin/env python
"""probe_real_data.py —— 真实金属伪影样本上的方向提取 + 方向滤波验证（探针第 1 步）。

在真实样本上复用 stripe_direction_probe 的算子：
  1. 结构张量提取条纹方向 θ + coherence（在 Baseline 观测图上）
  2. masked 方向低通（mask = 阴影区 d_j/d_ref < 9，保护充分区组织）
  3. 量化：方向滤波让观测更接近真值 Target（去掉条纹），且充分区几乎不变
  4. 存可视化图（x_hat / θ 场 / x̂_dir / 残差 / mask）+ 统计 JSON

用途：人工标定探针参数（σ 窗口、σ_par/σ_perp、mask 范围）。本脚本只读 Baseline/x_M0/d_j/
Target（Target 仅用于验证滤波「去伪影、保组织」，探针本身的 mask/θ 不依赖 Target，无泄漏）。

用法（服务器）：
    /root/miniconda3/envs/pzneuro/bin/python probe_real_data.py \
        --data-dir /root/autodl-tmp/PRISM/training_data \
        --out-dir /root/autodl-tmp/PRISM/dir_probe --n-samples 2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stripe_direction_probe import (  # noqa: E402
    structure_tensor, anisotropic_smooth, separable_smooth)

H = W = 512


def read_raw(path: str, h: int = H, w: int = W) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(h, w)


def find_metal_samples(phys_dir: str, frac_lo: float = 0.005, frac_hi: float = 0.15):
    """扫 Phys 找「有金属但未占满」的样本（x_M0 非零占比在 [frac_lo, frac_hi]）。"""
    hits = []
    for f in glob.glob(os.path.join(phys_dir, "*_x_M0.npy")):
        a = np.load(f)
        frac = float((a > 0).mean())
        if frac_lo <= frac <= frac_hi:
            hits.append((frac, os.path.basename(f).replace("_x_M0.npy", "")))
    hits.sort()
    return hits


def metal_neighbor_mask(x_M0: np.ndarray, radius: float = 20.0) -> np.ndarray:
    """距离金属 < radius 的体素 = 1（用高斯膨胀近似，无 scipy 依赖）。"""
    m = (x_M0 > 0).astype(np.float64)
    # 高斯平滑 σ=radius/3 → 金属邻域≈1，远处≈0；阈值 0.01 截断
    m_blur = separable_smooth(m, radius / 3.0)
    return (m_blur > 0.01).astype(np.float64)


def theta_rgb(theta: np.ndarray, coh: np.ndarray) -> np.ndarray:
    """θ 场可视化：hue=方向（θ/π），saturation=coherence。"""
    import matplotlib.colors as mcolors
    h = (theta / np.pi) % 1.0
    s = np.clip(coh, 0.0, 1.0)
    v = np.ones_like(h)
    hsv = np.stack([h, s, v], axis=-1)
    return mcolors.hsv_to_rgb(hsv)


def directional_filter_masked(x_hat, theta, coh, mask, sigma_par=1.0, sigma_perp=4.0):
    """masked 方向低通：只在 mask=1 处沿垂直条纹方向低通；mask=0 处原样。

    逐体素 θ 变化的近似：这里 θ 场可能逐体素变，但第 1 步先用「样本级主方向」验证；
    逐体素 steerable 滤波留到第 2 步落盘脚本。这里先验证「方向滤波对真实条纹有效」。
    """
    # 样本级主方向（金属邻域内 coherence 加权双倍角平均）
    m = (mask > 0.5) & (coh > 0.3)
    if m.sum() < 100:
        return x_hat.copy(), float("nan")
    th = theta[m]
    mean_cos2 = float(np.mean(np.cos(2.0 * th)))
    mean_sin2 = float(np.mean(np.sin(2.0 * th)))
    theta_main = 0.5 * np.arctan2(mean_sin2, mean_cos2) % np.pi

    x_dir = anisotropic_smooth(x_hat, theta_main, sigma_par, sigma_perp)
    out = mask * x_dir + (1.0 - mask) * x_hat
    return out, theta_main


def process_one(sample_id, train_dir, d_ref):
    base_dir = os.path.join(train_dir, "Baseline")
    tgt_dir = os.path.join(train_dir, "Target")
    phys_dir = os.path.join(train_dir, "Phys")

    x_hat = read_raw(os.path.join(base_dir, f"{sample_id}.raw"))
    target = read_raw(os.path.join(tgt_dir, f"{sample_id}.raw"))
    x_M0 = np.load(os.path.join(phys_dir, f"{sample_id}_x_M0.npy"))[0]
    d_j = np.load(os.path.join(phys_dir, f"{sample_id}_d_j.npy"))[0]

    # mask = 金属邻域 ∩ 阴影区（部分+完全阴影）：方向滤波只在「金属附近且观测不可信」处
    # 激活，充分区/远离金属的组织原样保护（第 0 步已验证 coherence 判别不可靠，必须金属邻域）。
    r = d_j / d_ref
    mask_metal = metal_neighbor_mask(x_M0, radius=20.0)
    mask_shadow = (r < 9.0).astype(np.float64)
    mask = mask_metal * mask_shadow

    # 结构张量提取 θ + coherence
    theta, coh = structure_tensor(x_hat, sigma=1.5)

    # masked 方向低通
    x_dir, theta_main = directional_filter_masked(x_hat, theta, coh, mask)

    # 统计：滤波前后 vs 真值 Target 的 RMSE（整体 / 金属邻域 / 远离金属）
    def rmse(a, b, sel=None):
        if sel is not None:
            a, b = a[sel], b[sel]
        return float(np.sqrt(np.mean((a - b) ** 2)))

    near = mask > 0.5
    far = ~near
    stats = {
        "sample_id": sample_id,
        "theta_main_deg": float(np.rad2deg(theta_main)),
        "metal_frac": float((x_M0 > 0).mean()),
        "mask_near_frac": float(near.mean()),
        "d_j_ratio_min/med/max": [float(r.min()), float(np.median(r)), float(r.max())],
        "rmse_before_total": rmse(x_hat, target),
        "rmse_after_total": rmse(x_dir, target),
        "rmse_before_near": rmse(x_hat, target, near),
        "rmse_after_near": rmse(x_dir, target, near),
        "rmse_before_far": rmse(x_hat, target, far),
        "rmse_after_far": rmse(x_dir, target, far),
    }
    return stats, {
        "x_hat": x_hat, "target": target, "theta": theta, "coh": coh,
        "x_dir": x_dir, "mask": mask, "resid": x_hat - x_dir, "x_M0": x_M0,
    }


def save_figs(sample_id, img, out_dir, theta_main_deg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    vmin, vmax = -1000, 1500

    ax = axes[0, 0]; ax.imshow(img["x_hat"], cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("x_hat (Baseline, artifact)"); ax.axis("off")
    ax = axes[0, 1]; ax.imshow(img["target"], cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Target (clean)"); ax.axis("off")
    ax = axes[0, 2]; ax.imshow(theta_rgb(img["theta"], img["coh"]))
    ax.set_title("theta field (hue=direction, sat=coherence)"); ax.axis("off")

    ax = axes[1, 0]; ax.imshow(img["x_dir"], cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("x_dir (masked directional-filtered)"); ax.axis("off")
    ax = axes[1, 1]; ax.imshow(img["resid"], cmap="gray", vmin=-400, vmax=400)
    ax.set_title("residual = x_hat - x_dir (removed streaks)"); ax.axis("off")
    ax = axes[1, 2]
    ax.imshow(img["x_M0"] > 0, cmap="Reds", alpha=0.6)
    ax.imshow(img["mask"], cmap="Blues", alpha=0.3)
    ax.set_title("mask (blue=shadow r<9, red=metal)"); ax.axis("off")

    fig.suptitle(f"{sample_id}  (theta_main={theta_main_deg:.0f}deg)")
    out_path = os.path.join(out_dir, f"{sample_id}_probe.png")
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data")
    ap.add_argument("--out-dir", default="/root/autodl-tmp/PRISM/dir_probe")
    ap.add_argument("--n-samples", type=int, default=2)
    args = ap.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    phys_dir = os.path.join(train_dir, "Phys")
    os.makedirs(args.out_dir, exist_ok=True)

    # d_ref（held-out 训练集冻结，同 B8）
    d_ref_path = os.path.join(args.data_dir, "d_ref_ho_patients_head2.npy")
    if os.path.exists(d_ref_path):
        d_ref = float(np.load(d_ref_path)[0])
    else:
        d_ref = float(np.load(os.path.join(args.data_dir, "d_ref.npy"))[0])
    print(f"[d_ref] {d_ref:.6g}", flush=True)

    samples = find_metal_samples(phys_dir, 0.02, 0.10)
    print(f"[find] {len(samples)} metal samples in [2%, 10%] range", flush=True)
    if not samples:
        samples = find_metal_samples(phys_dir, 0.01, 0.30)
        print(f"[find] relaxed: {len(samples)} samples", flush=True)

    all_stats = []
    for frac, sid in samples[: args.n_samples]:
        print(f"[proc] {sid} metal_frac={frac*100:.2f}%", flush=True)
        stats, img = process_one(sid, train_dir, d_ref)
        all_stats.append(stats)
        png = save_figs(sid, img, args.out_dir, stats["theta_main_deg"])
        print(f"[save] {png}", flush=True)
        print(f"[stat] {json.dumps(stats, ensure_ascii=False)}", flush=True)

    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"[done] {len(all_stats)} samples -> {args.out_dir}/stats.json", flush=True)


if __name__ == "__main__":
    main()
