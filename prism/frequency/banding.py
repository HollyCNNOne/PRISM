"""banding —— 分带观测精度的物理量（论文 §3.5 的实现）。

从原始径向 LFR 派生逐体素分带量（w_b 信息份额、τ_b^rad 每带信任、τ_eff 有效信任）：

    LFR_j(k)  = |Fℓ_j|(k) / |Fℓ_j|(0)               （DC 归一化幅度响应 ∈[0,1]）
    w_b       = LFR_b^rad / Σ_b' LFR_b'^rad          （信息份额，Σ_b w_b = 1）
    τ_b^rad   = φ_k^freq(LFR_b^rad)                   （每带信任，φ 阈值化）
    τ_eff     = Σ_b w_b τ_b^rad ∈ [0,1]               （有效信任）

φ_k^freq(t) = (t/d_ref_freq)^k / (1 + (t/d_ref_freq)^k)，d_ref_freq=0.5 固定（LFR 已
归一化，0.5 即「信号保留一半 → 信任跌到 0.5」），k=3（与空间 φ_k 同形）。

nb=1（标量退化）：单带平凡归一化 → w_b=1、τ_eff=τ_b^rad；响应平坦时 τ_eff=1。

分带常量（RADIAL_EDGES 等）复制自 AGIR 的 agir/lir_freq.py，本模块纯 numpy 自包含，
本机可自检。torch 侧不需要镜像——分带量是冻结物理输入（numpy 算好 → from_agir_lfr 转
torch），递归里唯一需要的 torch 运算 τ_eff = Σ_b w_b τ_b^rad 已内联在
models/m_network.py 的 forward 中。
"""

from __future__ import annotations

import numpy as np

# ---- 分带常量（复制自 AGIR/agir/lir_freq.py，保持一致）----
RADIAL_EDGES = np.array([0.002, 0.008, 0.032, 0.126, 0.5])  # 径向带边（对数间隔，4 带）
N_ANGULAR = 8                                                # 方向扇区数
D_REF_FREQ = 0.5                                             # φ_k^freq 转折点（固定）
K_FREQ = 3.0                                                 # 与空间 φ_k 同 k

EPS = 1e-12


def phi_freq(t, d_ref_freq: float = D_REF_FREQ, k: float = K_FREQ) -> np.ndarray:
    """φ_k^freq(t) = (t/d_ref)^k / (1 + (t/d_ref)^k)。t ∈ [0,1]（LFR 已归一化）。"""
    t = np.asarray(t, dtype=np.float64)
    r = (t / d_ref_freq) ** k
    return r / (1.0 + r)


def band_centers(r_edges: np.ndarray = RADIAL_EDGES, n_angular: int = N_ANGULAR):
    """每带代表频率 (R, TH)：R 取对数中心 sqrt(r_lo·r_hi)，TH 取扇区中心。

    返回 (R, TH)，各 shape (n_radial, n_angular)；R 单位 cyc/pixel，TH ∈ [0,π)。
    """
    r_lo = np.asarray(r_edges[:-1], dtype=np.float64)
    r_hi = np.asarray(r_edges[1:], dtype=np.float64)
    r_c = np.sqrt(r_lo * r_hi)
    th_c = (np.arange(n_angular) + 0.5) * (np.pi / n_angular)
    R, TH = np.meshgrid(r_c, th_c, indexing="ij")
    return R, TH


def radial_lfr_from_params(theta, smaj, smin, r_edges: np.ndarray = RADIAL_EDGES,
                           n_angular: int = N_ANGULAR) -> np.ndarray:
    """(θ, σ_maj, σ_min) 逐体素参数 → 原始径向 LFR_b^rad（角向坍缩）。

    每 (b,a) 格 LFR = 1/(1 + R_b²[cos²(TH_a−θ)/σ_maj² + sin²(TH_a−θ)/σ_min²])
    （比值模型，AGIR fit_anisotropic_lfr 的逆），再对 a 求平均得 LFR_b^rad。

    Parameters
    ----------
    theta/smaj/smin: (H, W) float，逐体素拟合参数（AGIR interpolate_lfr_params 输出）。

    Returns
    -------
    (n_radial, H, W) float，原始径向 LFR（未 φ 阈值化）。
    """
    R, TH = band_centers(r_edges, n_angular)          # (n_radial, n_angular)
    theta = np.asarray(theta, dtype=np.float64)
    smaj = np.asarray(smaj, dtype=np.float64)
    smin = np.asarray(smin, dtype=np.float64)
    n_r, n_a = R.shape
    acc = np.zeros((n_r,) + theta.shape)
    for ri in range(n_r):
        for ai in range(n_a):
            rr = R[ri, ai]
            dth = TH[ri, ai] - theta
            q = rr ** 2 * (np.cos(dth) ** 2 / smaj ** 2
                           + np.sin(dth) ** 2 / smin ** 2)
            acc[ri] += 1.0 / (1.0 + q)
    return acc / n_a


def radial_lfr_to_weights(lfr_radial, d_ref_freq: float = D_REF_FREQ,
                          k: float = K_FREQ):
    """原始径向 LFR → (w_b, τ_b^rad, τ_eff)。

    lfr_radial: (nb, ...)，band 在 axis 0。
    返回 w_b、τ_b^rad（各 (nb, ...)）、τ_eff（(1, ...)）。
    """
    lfr = np.asarray(lfr_radial, dtype=np.float64)
    w_b = lfr / (lfr.sum(axis=0, keepdims=True) + EPS)   # Σ_b w_b = 1
    tau_b = phi_freq(lfr, d_ref_freq, k)
    tau_eff = (w_b * tau_b).sum(axis=0, keepdims=True)   # ∈ [0,1]
    return w_b, tau_b, tau_eff


def from_agir_lfr(theta, smaj, smin, r_edges: np.ndarray = RADIAL_EDGES,
                  n_angular: int = N_ANGULAR):
    """AGIR (θ, σ_maj, σ_min) 逐体素 → torch 分带量 (tau_b, w_b)。

    整合封装：调本地 radial_lfr_from_params 重建原始径向 LFR → radial_lfr_to_weights
    派生分带量 → 转 torch。torch 延迟导入（本机无 torch 也能跑 numpy 自检）。

    Parameters
    ----------
    theta/smaj/smin: (H, W) numpy（AGIR interpolate_lfr_params 输出）。

    Returns
    -------
    tau_b, w_b: 各 (1, nb, H, W) torch float32（冻结输入，无梯度）。
    """
    import torch

    lfr = radial_lfr_from_params(theta, smaj, smin, r_edges, n_angular)  # (nb, H, W)
    w_b, tau_b, _ = radial_lfr_to_weights(lfr)
    tau_b = torch.from_numpy(tau_b).float().unsqueeze(0)
    w_b = torch.from_numpy(w_b).float().unsqueeze(0)
    return tau_b, w_b


def _self_test() -> bool:
    """本机纯 numpy 自检：φ 范围/单调、w_b 归一化、τ_eff 范围、nb=1 退化、平坦退化。"""
    print("[banding self-test]")
    ok = True

    # Test 1：φ_k^freq 范围 ∈[0,1]，单调增，φ(d_ref)=0.5
    t = np.linspace(0.0, 1.0, 1001)
    phi = phi_freq(t)
    t1 = float(phi.min()) >= 0.0 and float(phi.max()) <= 1.0
    t1 &= bool(np.all(np.diff(phi) >= 0.0))
    t1 &= abs(float(phi_freq(D_REF_FREQ)) - 0.5) < 1e-9
    ok &= t1
    print(f"  [1] phi range/monotone/phi(0.5)=0.5 -> {'OK' if t1 else 'FAIL'}")

    # Test 2：w_b 归一化和为 1；τ_eff ∈[0,1]
    lfr = np.array([0.9, 0.6, 0.3, 0.1])[:, None, None]
    w_b, tau_b, tau_eff = radial_lfr_to_weights(lfr)
    t2 = abs(float(w_b.sum()) - 1.0) < 1e-9
    t2 &= float(tau_eff.min()) >= 0.0 and float(tau_eff.max()) <= 1.0
    ok &= t2
    print(f"  [2] w_b sums to 1 / tau_eff in [0,1] "
          f"(tau_eff={float(tau_eff[0, 0, 0]):.3f}) -> {'OK' if t2 else 'FAIL'}")

    # Test 3：nb=1 退化 —— 单带 w_b=1、τ_eff=τ_b^rad
    lfr1 = np.array([0.7])[:, None, None]
    w1, tau1, teff1 = radial_lfr_to_weights(lfr1)
    t3 = abs(float(w1[0, 0, 0]) - 1.0) < 1e-9
    t3 &= abs(float(teff1[0, 0, 0]) - float(tau1[0, 0, 0])) < 1e-9
    ok &= t3
    print(f"  [3] nb=1 degenerate (w_b=1, tau_eff=tau_b) -> {'OK' if t3 else 'FAIL'}")

    # Test 4：平坦响应 → 各带同 τ、τ_eff = φ(flat)
    lfr_flat = np.full((4, 3, 3), 0.8)
    w4, tau4, teff4 = radial_lfr_to_weights(lfr_flat)
    t4 = np.allclose(w4, 1.0 / 4.0) and np.allclose(teff4, phi_freq(0.8))
    ok &= t4
    print(f"  [4] flat response -> w_b=1/nb, tau_eff=phi(flat) -> {'OK' if t4 else 'FAIL'}")

    # Test 5：radial_lfr_from_params 输出形状 + 原始 LFR 范围（各向同性参数）
    theta = np.full((8, 8), np.deg2rad(30.0))
    smaj = np.full((8, 8), 0.10)
    smin = np.full((8, 8), 0.03)
    lfr_rad = radial_lfr_from_params(theta, smaj, smin)
    t5 = lfr_rad.shape == (4, 8, 8)
    t5 &= float(lfr_rad.min()) >= 0.0 and float(lfr_rad.max()) <= 1.0
    ok &= t5
    print(f"  [5] radial_lfr_from_params shape/range -> {'OK' if t5 else 'FAIL'}")

    print(f"  => {'ALL OK' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _self_test() else 1)
