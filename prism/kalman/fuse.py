"""fuse —— 双向融合（前向趟 + 后向趟）。

精度逐带相加、均值用「总精度」（带和）加权、速度不融合：
    p_post^b = p1^b + p2^b
    μ = (Σ_b p1^b·μ1 + Σ_b p2^b·μ2) / (Σ_b p1^b + Σ_b p2^b)
一方精度为 0 → 退化为另一方。返回 KalmanState（delta_mu=None，速度不融合）。
"""

from __future__ import annotations

import torch

from .state import KalmanState, total_precision

EPS = 1e-8


def fuse(state1, state2):
    """精度加权融合两个方向的递归态。

    μ 是标量场 [B, 1, H, W]，用总精度（带和 [B, 1, H, W]）加权；p_post 逐带相加 [B, nb, H, W]。
    """
    p1 = state1.p_post
    p2 = state2.p_post
    p_sum = p1 + p2                                   # [B, nb, H, W] 逐带
    tp1 = total_precision(p1)                          # [B, 1, H, W] 带和
    tp2 = total_precision(p2)
    p_tot = tp1 + tp2
    mu = (tp1 * state1.mu + tp2 * state2.mu) / (p_tot + EPS)
    return KalmanState(mu=mu, p_post=p_sum, delta_mu=None)
