"""update —— 卡尔曼更新步（补）。

分带更新：p_post^b = p_pred^b + τ_b^rad · w_b · d_j/d_ref
   τ_b^rad、w_b 是逐体素分带量 [B, nb, H, W]（物理 LFR 派生，见 frequency/banding.py）；
   d_j 是标量场 [B, 1, H, W] 广播到带维。τ_b^rad、w_b 为 None 时退化为标量
   （τ=w=1 → p_post = p_pred + d_j/d_ref，即 nb=1 的标量形式）。

C4：p_post clip 到 [1e-6, 1e6]。
约束（5.1 #7）：d_j 只在这里进入递归。
C5：d_ref 是训练集冻结的标量（median(d_j)），由调用方传入，本模块不重算。
C6（B8）：观测项乘阴影感知调制 s（见 gain.shadow_obs_scale）——完全阴影里观测不可信，
   既不进均值（K→1）也不进精度（obs·s→0），否则后验会从一个打穿的观测「假增信心」。
"""

from __future__ import annotations

import torch

from .gain import shadow_obs_scale

P_MIN = 1e-6
P_MAX = 1e6


def update_precision(p_pred, d_j, d_ref, tau_b=None, w_b=None, s=None):
    """后验精度 = 预测精度 + 分带观测精度，并 clip（C4）。

    tau_b、w_b 为 None → 标量退化（观测项 = d_j/d_ref）。
    否则观测项 = τ_b^rad · w_b · d_j/d_ref，逐带 [B, nb, H, W]。
    s 是阴影感知调制（B8）；None → 不调制（s≡1，基座公式），由 m_network 显式传入。
    """
    obs = d_j / d_ref
    if tau_b is not None and w_b is not None:
        obs = tau_b * w_b * obs
    if s is not None:
        obs = obs * s
    p = p_pred + obs
    return torch.clamp(p, min=P_MIN, max=P_MAX)
