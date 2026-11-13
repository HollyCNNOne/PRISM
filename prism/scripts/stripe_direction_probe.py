#!/usr/bin/env python
"""stripe_direction_probe.py —— 条纹方向提取 + 方向性观测预处理（第 0 步自检）。

「方向性信任」探针的物理前提验证，纯 numpy、本机可跑、零服务器：

  前提 1：条纹方向 θ 能否从观测图 x_hat 的结构张量可靠提取（无泄漏、推理可得）。
  前提 2：沿「垂直条纹方向」低通，能否抑制条纹伪影、同时保留「沿条纹方向」的组织结构。

背景（为什么这么做）：
  B8 在部分阴影区 K→0 全信观测 x̂，但 x̂ 里「垂直条纹方向」的分量是金属伪影的确定性
  bias，被全信后 carry 进完全阴影 → 完全阴影 bias +77（vs 纯先验地板 +5）、giant carry
  污染 +447。方向性信任 = 在观测进入卡尔曼修正前，先沿垂直条纹方向低通，把条纹伪影
  滤掉，再 μ_post = K·μ_pred + (1−K)·x̂_dir。

约定（与 AGIR lir_freq 一致）：θ 指频域扇区方向；空间条纹方向与之垂直。本脚本的
  θ_stripe 是**空间条纹方向**（结构张量变化最小的方向 e2）；沿其垂直方向做低通。
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- 基础算子

def gaussian_kernel_1d(sigma: float, radius: int | None = None) -> np.ndarray:
    """一维高斯核（separable 平滑用）。radius 缺省取 ceil(3σ)。"""
    if radius is None:
        radius = int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def separable_smooth(img: np.ndarray, sigma: float) -> np.ndarray:
    """separable 高斯平滑（numpy 卷积，无 scipy 依赖）。"""
    k = gaussian_kernel_1d(sigma)
    out = img.astype(np.float64)
    for axis in (0, 1):
        out = np.apply_along_axis(
            lambda v: np.convolve(v, k, mode="same"), axis, out)
    return out


def structure_tensor(img: np.ndarray, sigma: float = 1.5):
    """结构张量 → (theta_stripe, coherence)。

    theta_stripe: (H,W) 空间条纹方向（灰度变化最小的方向 e2），∈ [0, π)。
    coherence:    (H,W) (λ1−λ2)/(λ1+λ2) ∈ [0,1]，1=强方向性（条纹），0=各向同性（无条纹）。
    """
    gy, gx = np.gradient(img.astype(np.float64))
    J11 = separable_smooth(gx * gx, sigma)
    J12 = separable_smooth(gx * gy, sigma)
    J22 = separable_smooth(gy * gy, sigma)

    # 2×2 对称矩阵特征分解（闭式）
    tr = J11 + J22
    disc = np.sqrt(np.maximum((J11 - J22) ** 2 + 4.0 * J12 ** 2, 0.0))
    lam1 = 0.5 * (tr + disc)   # 变化最大方向（垂直条纹）
    lam2 = 0.5 * (tr - disc)   # 变化最小方向（条纹方向）

    # 变化最大方向 e1 的角度（0.5·atan2(2J12, J11−J22)）；条纹方向 = e1 + π/2
    theta_e1 = 0.5 * np.arctan2(2.0 * J12, J11 - J22)
    theta_stripe = (theta_e1 + np.pi / 2.0) % np.pi

    denom = lam1 + lam2
    coherence = np.where(denom > 1e-12, (lam1 - lam2) / np.maximum(denom, 1e-12), 0.0)
    return theta_stripe, coherence


def anisotropic_gaussian_kernel(theta: float, sigma_par: float, sigma_perp: float,
                                radius: int = 12) -> np.ndarray:
    """各向异性高斯核：长轴沿 theta（σ_par 小，保留），短轴垂直 theta（σ_perp 大，抹平）。

    g(p) = exp(−( (p·t̂)²/(2σ_par²) + (p·n̂)²/(2σ_perp²) ))，t̂=(cosθ,sinθ), n̂=(−sinθ,cosθ)。
    """
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float64)
    ct, st = np.cos(theta), np.sin(theta)
    p_t = xx * ct + yy * st          # 沿 theta 分量（条纹方向）
    p_n = -xx * st + yy * ct         # 垂直 theta 分量（条纹变化方向）
    k = np.exp(-(p_t ** 2) / (2.0 * sigma_par ** 2)
               - (p_n ** 2) / (2.0 * sigma_perp ** 2))
    return k / k.sum()


def anisotropic_smooth(img: np.ndarray, theta: float, sigma_par: float,
                       sigma_perp: float) -> np.ndarray:
    """固定方向各向异性平滑（FFT 卷积，纯 numpy）。"""
    k = anisotropic_gaussian_kernel(theta, sigma_par, sigma_perp)
    h, w = img.shape
    # 零填充 FFT 卷积
    pad_h, pad_w = k.shape[0] // 2, k.shape[1] // 2
    img_p = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    fft_shape = np.array(img_p.shape) + np.array(k.shape) - 1
    F = np.fft.rfft2(img_p, s=fft_shape)
    K = np.fft.rfft2(k, s=fft_shape)
    out = np.fft.irfft2(F * K, s=fft_shape).real
    out = out[pad_h:pad_h + h, pad_w:pad_w + w]
    return out


# ---------------------------------------------------------------- 自检

def _synthetic(shape=(160, 160), theta_true_deg=30.0, lam=8.0, amp=80.0,
               x_split=80, sigma_org=10.0, seed=0):
    """合成图：左半 = 组织 + 沿 θ_true 的正弦条纹（模拟金属附近）；右半 = 纯组织（模拟充分区）。

    组织底图 σ_org=10px（尺度 > 条纹周期 8px），对应真实「颅脑组织平滑、金属条纹高频」——
    方向低通才可分离：抹细条纹、留粗组织。
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    # 组织底图：低通高斯随机场（各向同性，尺度 σ_org）
    org = rng.standard_normal(shape)
    org = separable_smooth(org, sigma_org)
    org = org / (org.std() + 1e-12) * 20.0  # 组织波动 ~20 HU

    # 条纹：沿单位向量 t̂=(cosθ, sinθ) 灰度不变，沿垂直 n̂ 正弦变化
    theta = np.deg2rad(theta_true_deg)
    nt_x, nt_y = -np.sin(theta), np.cos(theta)   # 垂直条纹方向 n̂
    streak = amp * np.sin(2.0 * np.pi * (xx * nt_x + yy * nt_y) / lam)
    streak = streak * (xx < x_split)              # 只在左半（金属附近）注入条纹

    img = org + streak
    return img, org, streak


def _dir_grad_energy(img: np.ndarray, angle_deg: float) -> float:
    """沿 angle 方向的梯度能量 = mean(|∇img · n̂(angle)|²)。"""
    gy, gx = np.gradient(img.astype(np.float64))
    a = np.deg2rad(angle_deg)
    return float(np.mean((gx * np.cos(a) + gy * np.sin(a)) ** 2))


def _self_test() -> bool:
    print("[stripe_direction_probe self-test]")
    ok = True
    theta_true = 30.0
    img, org, streak = _synthetic(theta_true_deg=theta_true)

    # ---- 前提 1：θ 提取 ----
    theta_est, coh = structure_tensor(img, sigma=1.5)
    # 条纹区（左半中心带）的 θ 估计，按 coherence 高的体素统计
    mask = np.zeros_like(coh, dtype=bool)
    mask[:80, 10:70] = True                      # 左半中心（条纹区）
    mask &= coh > 0.5                            # 只取高方向性体素
    # 双倍角平均（线方向 θ↔θ+π 不变）
    if mask.sum() > 0:
        th = theta_est[mask]
        mean_cos2 = float(np.mean(np.cos(2.0 * th)))
        mean_sin2 = float(np.mean(np.sin(2.0 * th)))
        theta_hat = 0.5 * np.arctan2(mean_sin2, mean_cos2) % np.pi
        err = np.rad2deg(abs(((theta_hat - np.deg2rad(theta_true) + np.pi / 2) % np.pi) - np.pi / 2))
    else:
        theta_hat, err = float("nan"), 999.0
    t1 = err < 5.0
    ok &= t1
    print(f"  [1] theta extraction: true={theta_true}deg est={np.rad2deg(theta_hat):.1f}deg "
          f"err={err:.2f}deg (n={mask.sum()}) -> {'OK' if t1 else 'FAIL'}")

    # coherence 不能可靠区分条纹 vs 平滑组织：大尺度组织在小窗口里 coherence 也高。
    # 这正说明「滤波范围必须用金属邻域 mask，不能靠 coherence 判别」。断言充分区 coherence
    # 依然高（>0.5），即「coherence 判别不可靠」这一负面事实成立。
    coh_streak = coh[:80, 10:70].mean()
    coh_right = coh[:, 90:].mean()
    t2 = coh_right > 0.5
    ok &= t2
    print(f"  [2] coherence unreliable: streak={coh_streak:.3f} vs plain={coh_right:.3f} "
          f"(plain>0.5 -> mask, not coherence) -> {'OK' if t2 else 'FAIL'}")

    # ---- 前提 2：方向性低通 ----
    # 条纹周期 λ=8px → σ_perp 取 λ/2 量级才能抹平；σ_par 取小保留条纹方向组织。
    sigma_par, sigma_perp = 1.0, 4.0
    img_dir = anisotropic_smooth(img, np.deg2rad(theta_true), sigma_par, sigma_perp)

    # 垂直条纹方向（120°）梯度能量：滤波前大（条纹），滤波后应大幅下降
    e_before_perp = _dir_grad_energy(img, theta_true + 90.0)
    e_after_perp = _dir_grad_energy(img_dir, theta_true + 90.0)
    drop = (e_before_perp - e_after_perp) / max(e_before_perp, 1e-12)
    t3 = drop > 0.5
    ok &= t3
    print(f"  [3] perp-dir grad energy: {e_before_perp:.1f} -> {e_after_perp:.1f} "
          f"(drop={drop*100:.1f}%) -> {'OK' if t3 else 'FAIL'}")

    # 平行条纹方向（30°）梯度能量：组织保留，滤波前后不应大幅下降
    e_before_par = _dir_grad_energy(img, theta_true)
    e_after_par = _dir_grad_energy(img_dir, theta_true)
    keep = e_after_par / max(e_before_par, 1e-12)
    t4 = keep > 0.4
    ok &= t4
    print(f"  [4] par-dir grad energy: {e_before_par:.1f} -> {e_after_par:.1f} "
          f"(keep={keep*100:.1f}%) -> {'OK' if t4 else 'FAIL'}")

    # ---- 限定范围（masked filtering）：只在条纹区（金属侧）滤波，充分区原样 ----
    # 这是探针的核心保护机制：coherence 不可靠（组织边缘也高），滤波范围必须用
    # 金属邻域 mask 限定，否则充分区的组织会被误伤。
    h, w = img.shape
    _yy, xx = np.mgrid[0:h, 0:w]
    x_split = 80
    side_mask = (xx < x_split).astype(np.float64)        # 金属侧 = 1，充分区 = 0
    img_masked = side_mask * img_dir + (1.0 - side_mask) * img
    # 充分区（mask=0）必须原样：rmse 相对组织 std 应极小
    rmse_plain = float(np.sqrt(np.mean((img_masked[:, 90:] - img[:, 90:]) ** 2)))
    org_std = float(np.sqrt(np.mean(org ** 2)))
    t5 = rmse_plain < 1e-6
    ok &= t5
    # 条纹区（mask=1）应被滤波（与滤波后的 img_dir 一致）
    t5b = np.allclose(img_masked[:, :x_split], img_dir[:, :x_split], atol=1e-6)
    ok &= t5b
    print(f"  [5] masked filtering: plain-region rmse={rmse_plain:.2e} (org_std={org_std:.1f}) "
          f"/ streak-region filtered -> {'OK' if t5 and t5b else 'FAIL'}")

    print(f"  => {'ALL OK' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _self_test() else 1)
