#!/usr/bin/env python3
"""parallel_beam.py — 解析管线（skimage 平行束）的单一物理真源。

Scope: ANALYTIC pipeline ONLY（radon/iradon 自包含）。不触碰任何 catsim/xcist
全物理函数。所有解析重生成脚本（regen_skimage.py / regen_phys.py）都从本模块
导入 `simulate_sino`，保证 Baseline / LI / d_j / g 四者共享**同一噪声实现、
同一随机种子、同一 rng 调用顺序**——d_j 的权重 W 必须建立在 Baseline 重建
实际吃到的那条测量 sinogram 上，否则数据充分度与观测图不一致。

物理管线（与探针 render_analytic_mar.py 定案版逐行一致）：

    L_tissue = radon((1-mask)·hu2mu(clean))·PIXEL_MM   (组织线积分)
    L_metal  = radon(mask·METAL_MU)·PIXEL_MM           (金属线积分，μ=7.02 mm⁻¹)
    L_tissue ← L_tissue·(1 − β_bh·L_tissue)            (束硬化杯突，仅组织)
    L = L_tissue + L_metal
    I = I0·exp(−L)
    I_noisy = poisson(I) + N(0, σ_e)                   (光子 + 电子噪声地板)
    I_noisy = max(I_noisy, 0.5)
    L_noisy = −log(I_noisy/I0)
    L_noisy ← L_noisy + N(0, σ_starve)·1{L > L_starve} (光子饥饿：阴影区直接 L 域强噪声)
    L_noisy ← gaussian_filter1d(L_noisy, σ=PSF[size])  (尺寸相关探测器 PSF 模糊)

随机种子 = int(sample_id 后缀)，nproc 顺序无关（确定性）。
"""
from __future__ import annotations

import numpy as np
from skimage.transform import radon

MU_WATER = 0.02
METAL_MU = 7.02            # mm⁻¹ = 350000 HU (Pt-W)
PIXEL_MM = 220.16 / 512

# 尺寸相关探测器 PSF σ（sinogram 探测器轴高斯模糊，单位=探测器像素）。
PSF_BY_SIZE = {"small": 6.0, "medium": 6.0, "large": 6.0, "giant": 6.0}

I0 = 1e5
N_VIEWS = 720
SIGMA_E = 20.0             # 电子噪声地板（counts）；对数压缩 → 只定阴影暗度，斑驳靠 SIGMA_STARVE
BETA_BH = 1.5e-2           # 束硬化系数（仅组织，杯突）；0 = 关
L_STARVE = 6.0             # 线积分超过此值 = 探测器光子饥饿
SIGMA_STARVE = 3.0         # 光子饥饿区直接注入的 L 域噪声（线性，可控 → 斑驳暗带）

BASELINE_FILTER = "hann"
LI_FILTER = "shepp-logan"
LO, HI = -1000.0, 3000.0

THETA = np.linspace(0.0, 180.0, N_VIEWS, endpoint=False)


def hu_to_mu(hu):
    return MU_WATER * (1.0 + hu / 1000.0)


def mu_to_hu(mu):
    return (mu / MU_WATER - 1.0) * 1000.0


def simulate_sino(clean, mask, cat, seed):
    """前投 + 噪声 → 最终测量 sinogram L_noisy（PSF 模糊后）。

    clean/mask: (512,512) float32（HU / 体积分数）。cat: size key。seed: int。
    返回 L_noisy：(N_VIEWS, ~729) float64 线积分 sinogram（与 radon(circle=True) 同形）。

    rng 调用顺序固定为 poisson → σ_e → σ_starve 三次，任何消费者都必须原样复制，
    否则同 seed 下产出的 L_noisy 不同（d_j 与 Baseline 错位）。
    """
    from scipy.ndimage import gaussian_filter1d

    rng = np.random.RandomState(seed)

    tissue_mu = (1.0 - mask) * hu_to_mu(clean)
    metal_mu = mask * METAL_MU
    L_tissue = radon(tissue_mu, theta=THETA, circle=True) * PIXEL_MM
    L_metal = radon(metal_mu, theta=THETA, circle=True) * PIXEL_MM
    # 束硬化（仅组织）：次线性 → 杯突
    if BETA_BH > 0:
        L_tissue = L_tissue * (1.0 - BETA_BH * L_tissue)
    L = L_tissue + L_metal

    I = I0 * np.exp(-L)
    I_noisy = rng.poisson(I)                          # rng call #1
    I_noisy = I_noisy + rng.normal(0.0, SIGMA_E, I_noisy.shape)   # rng call #2
    I_noisy = np.maximum(I_noisy, 0.5)
    L_noisy = -np.log(I_noisy / I0)
    # 光子饥饿：探测器动态范围之外，阴影线积分本身就是噪声
    if SIGMA_STARVE > 0:
        L_noisy = L_noisy + rng.normal(0.0, SIGMA_STARVE, L.shape) * (L > L_STARVE)  # #3
    sigma = PSF_BY_SIZE.get(cat, 2.0)
    # PSF 模糊轴定案 axis=1（角度轴）：视觉上最接近临床「脏、糊」星芒；axis=0（探测器轴）
    # 会径向过糊，V5 已判负。别再改成 axis=0。
    L_noisy = gaussian_filter1d(L_noisy, sigma=sigma, axis=1, mode="nearest")
    return L_noisy
