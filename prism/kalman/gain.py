"""gain —— 卡尔曼增益（定义式，不用稳态闭式）。

分带增益（恒标量）：K = Σ_b p_pred^b / (Σ_b p_pred^b + τ_eff · d_j/d_ref · s)
   τ_eff = Σ_b w_b τ_b^rad ∈ [0, 1]（由 frequency/banding.py 算好传入）。
   带结构住在精度里（逐带传播），增益到达 FIND-Net 的是标量 K。
   响应平坦时 τ_eff=1 → 退化为标量比 K = p_pred/(p_pred + d_j/d_ref · s)。

语义：K 是「信先验」的权重。d_j→0（阴影）→ K→1；d_j 大（充分）→ K→0。

B8（阴影感知观测精度）：金属伪影是确定性 bias，不是光子噪声，观测在完全阴影里不可信；
   d_j/d_ref 只反映光子数、不反映「这条线穿过金属」。于是观测精度乘一个按 r=d_j/d_ref
   的尖锐门 s（见 shadow_obs_scale）：r<r0（完全阴影）→ s→0 → K→1 信先验；
   r>r0（部分阴影）→ s→1 → K→0 信观测。
"""

from __future__ import annotations

import torch

from .state import total_precision

EPS = 1e-8

SHADOW_R0 = 1.0 / 9.0          # 完全/部分阴影边界（与 eval 三分区一致：r < 1/9 为完全阴影）


def shadow_obs_scale(d_j, d_ref, r0=SHADOW_R0):
    """阴影感知观测精度调制 s ∈ {0,1}（B8：修「K 在完全阴影不尖 → 漏打穿的观测」）。

    硬阶跃：s = 1{r ≥ r0}，r = d_j/d_ref。
      r < r0（完全阴影）→ s=0 → 观测精度精确为 0 → K=1（信先验、别漏打穿的观测）；
      r ≥ r0（部分阴影 + 充分）→ s=1 → 观测精度不变 → K→0（信观测、保细节）。

    为什么必须硬阶跃而不是 sigmoid：Q 被监督到真实解剖波动（大）→ 先验精度 p_sum 很小
    （~0.01~0.03）。K = p_sum/(p_sum + r·s) 里，只要 s>0（sigmoid 永远 s>0）就有 r·s 与
    p_sum 同量级、K 到不了 1 → 打穿的观测漏进 μ。硬阶跃 s=0 使 K 精确=1。
    s 无学习参数、对 r 不求导（d_j 是冻结输入），硬阶跃不产生梯度问题；K 对 p_sum（网络输出）
    仍是光滑的，反向正常。
    """
    r = d_j / d_ref
    return (r >= r0).to(dtype=d_j.dtype)


def kalman_gain(p_pred, d_j, d_ref, tau_eff=None, s=None):
    """K = Σ_b p_pred^b / (Σ_b p_pred^b + τ_eff·d_j/d_ref·s + ε)。

    p_pred [B, nb, H, W] → Σ_b 得 [B, 1, H, W]；tau_eff None → 1（标量退化）。
    s 是阴影感知调制（B8）；None → 不调制（s≡1，基座公式，单元测试语义不变），
    由 m_network 按 shadow_obs_scale(d_j, d_ref) 显式传入。
    """
    p_sum = total_precision(p_pred)                     # [B, 1, H, W] 带和
    tau = 1.0 if tau_eff is None else tau_eff
    obs = d_j / d_ref
    if s is not None:
        obs = obs * s
    return p_sum / (p_sum + tau * obs + EPS)
