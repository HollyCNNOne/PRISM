"""predict —— 卡尔曼预测步（推）的解析部分。

μ_z     = (1 − Δ_z)·μ_{z-1} + Δ_z·μ_candidate
p_pred^b = p_post_{z-1}^b / ((1 − Δ_z)² + Q_z·p_post_{z-1}^b + ε)

分带：p_post 为 [B, nb, H, W]，逐带独立传播；Δ_z、Q_z 是跨带共享的标量场 [B, 1, H, W]，
    广播到带维——带间不混合，一个带不可分辨就单独保持不确定，不污染别的带。
C4：p_pred 分母加 ε = 1e-8。
约束（5.1 #7）：解析预测步不碰 d_j —— d_j 只进解析更新步 update.py。
"""

from __future__ import annotations

import torch

EPS = 1e-8


def predict_mean(mu_prev, delta, mu_candidate):
    """预测均值。μ_z = (1 − Δ_z)·μ_{z-1} + Δ_z·M_μ。"""
    return (1.0 - delta) * mu_prev + delta * mu_candidate


def predict_precision(p_post_prev, delta, q, eps=EPS):
    """预测精度。p_pred^b = p_post^b / ((1 − Δ_z)² + Q_z·p_post^b + ε)。

    p_post_prev [B, nb, H, W]；delta、q [B, 1, H, W]（跨带共享，广播）。
    """
    denom = (1.0 - delta) ** 2 + q * p_post_prev + eps
    return p_post_prev / denom
