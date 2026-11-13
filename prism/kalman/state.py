"""KalmanState —— 卡尔曼递归的状态定义。

C1：delta_mu 存「先验速度」 Δμ_z = μ_pred,z − μ_pred,{z-1}（B9 起；之前存后验速度），
    h_z 读上一片的先验速度。μ、Δμ 为逐体素场 [B, 1, H, W]。
分带：p_post 为逐体素分带精度场 [B, nb, H, W]（nb=1 标量退化，nb=4 分带默认）。
C6：cold_start 造 S_0 = (μ=0, p_post=0[nb], Δμ=0)。

B9（输出 vs carry 解耦，**只解耦均值、不解耦精度**）：μ 是「输出」——观测修正后的后验，
供下游 R_θ/融合消费；mu_carry 是「carry」——预测步的先验均值 μ_pred，供递归读、沿 z 传播，
不被观测污染。精度 p_post 保持 B8 不变（后验精度 = 预测精度 + 观测精度，**继续进 carry**）：
它是「确定性」的量，决定 K，不能被去掉观测项——否则 p_pred 从 0 永远长不起来、K 恒 0、
完全阴影后验=打穿的观测（B9 初版 bug：误把精度也解耦成 p_carry=p_pred，p_pred 卡 0）。
这拆掉 B8 的病根：部分阴影 K→0 时后验=观测 x̂，若把它当 carry 传进完全阴影，carry 就是
污染的观测（bias +77），而不是干净先验（B6 地板 +5）。mu_carry 缺省 None 向后兼容
（融合态、测试构造的干净态无需 carry）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class KalmanState:
    mu: torch.Tensor          # 后验均值 μ_post（观测修正输出，供 R_θ），[B, 1, H, W]
    p_post: torch.Tensor      # 后验精度（分带，carry + 输出，供 trust / R_θ），[B, nb, H, W]
    delta_mu: torch.Tensor    # 先验速度 Δμ_z = μ_pred,z − μ_pred,{z-1}（C1），[B, 1, H, W]
    mu_carry: torch.Tensor = None   # B9：先验均值 μ_pred（递归 carry，不被观测污染）

    @classmethod
    def cold_start(cls, size, device="cpu", dtype=torch.float32, nb=4):
        """冷启动 S_0（C6）。size = (B, H, W)；nb = 分带数（1=标量退化）。"""
        B, H, W = size
        mu = torch.zeros((B, 1, H, W), device=device, dtype=dtype)
        p_post = torch.zeros((B, nb, H, W), device=device, dtype=dtype)
        return cls(mu=mu, p_post=p_post, delta_mu=mu, mu_carry=mu)


def total_precision(p_post):
    """分带精度 → 总精度（带和）。p_post [B, nb, H, W] → [B, 1, H, W]。

    μ 是标量场，其融合权重与增益都取总精度 Σ_b p^b（§3.5 增益、§3.6 融合共用）；
    带结构仍保留在 p_post 里逐带传播，此处只做带维塌缩。nb=1 时退化为恒等。
    """
    return p_post.sum(dim=1, keepdim=True)
