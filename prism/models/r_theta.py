"""r_theta —— 下游 FIND-Net R_θ 的可微替身（end-to-end 训练用）。

v9 §3.7：最终重建 = 先验 μ + 增益加权拉拽（观测修正）。真实 FIND-Net 是独立仓库的
多阶段去除网络（FE-ResNet + GFFC，10 stages）；本模块是它的**最小可微替身**，唯一目的
是让下游重建损失 L_down = ‖I⊙(X − x̂_final)‖² 的梯度能穿过它回流到 SSM 的 M_μ（经 μ
输入）与 Q_z（经 p_post 输入），从而在 FIND-Net 正式接入前就能 end-to-end 训 SSM。

残差形式（关键）：x̂_final = μ + R_θ([x̂ − μ, μ, p_post_sum, d_j/d_ref])。
末层小权重初始化 → 初始 x̂_final ≈ μ → μ 首先被 L_down 拉向干净组织（学结构），随后
R_θ 学「观测修正」（去伪影 + 补细节）。R_θ 用慢学习率（M_μ 的 1/10）保住小权重护盾，
防止 R_θ 快速生长后吸走全部结构、μ 退化成平坦先验（lr=1e-3 时 corr 掉到 0 的根因）。

输入 4 通道（逐片 2D，FIND-Net 是纯逐片无 z）：
    1. x̂ − μ       观测与先验之差（伪影估计），HU → kHU
    2. μ           先验均值，HU → kHU
    3. trust       R_θ 的信任信号（C 消融 H2：kalman=p_post_sum / gated=π=g⊙c / fixed=γ）
    4. d_j/d_ref   物理数据充分度（门控信号，冻结输入但参与组合）
"""

from __future__ import annotations

import torch
import torch.nn as nn

# HU 归一化尺度：μ / x̂ 是 HU 量级（~±2000），p_post_sum / d_j/d_ref 是 O(1)。
# 用 100（100 HU = 1 单位）而非 1000：残差/100 量级 ~6，平方 ~40，梯度 2(μ−X)/1e4 ~ 0.1
# ——够 μ 在几十 epoch 内学到组织量级。用 1000 会得梯度 ~2e-3，μ 学不动（旧 μ 损失躺平的根因）。
HU_SCALE = 100.0
_EPS = 1e-8


class RTheta(nn.Module):
    """下游去除替身：先验 + 观测 → 最终重建 x̂_final = μ + corr。"""

    def __init__(self, channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 3, padding=1),
        )
        # 末层小权重 → 初始 corr ≈ 0 → x̂_final ≈ μ（先学干净先验，再学观测修正）
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_hat, mu, trust, d_j, d_ref, metal_mask=None):
        """x_hat/mu/trust/d_j: [B, 1, Z, H, W]；返回 x̂_final [B, 1, Z, H, W]。

        metal_mask: [B, 1, Z, H, W]（1=金属，0=组织），可选。非 None 时 corr 只在金属区
        作用：x̂_final = μ + corr·metal_mask，非金属区 x̂_final ≡ μ。这拆掉「x̂_final=μ+corr
        里 corr 全图无约束 → corr 吸走全部信号、μ 退化」的病根（B4/B5 加 L_mu 仍崩的原因）：
        非金属区 L_down 直接 ≡ L_mu，μ 被逼成干净组织，corr 只负责金属 inpaint——与真实
        FIND-Net（金属掩膜驱动去除网络）一致。金属区无干净 GT（L_down/L_corr 都掩到非金属），
        corr 在金属区无梯度，等效禁用；SSM 消融测的就是 μ/递归本身。

        逐片 2D：把 Z 维并入 batch（[B,1,Z,H,W] → [B*Z,4,H,W]），共享同一 conv。
        """
        B, _, Z, H, W = mu.shape
        d_j_n = d_j / d_ref

        def flat(t):
            return t.permute(0, 2, 1, 3, 4).reshape(B * Z, 1, H, W)

        # μ 进 R_θ 输入时 detach：R_θ 只把 μ 当「固定先验参考」读（算残差方向），
        # 梯度不回传给 M_μ。否则 corr_net = R_θ(μ) 依赖 μ，ridge 惩罚对 μ 的梯度是
        # 2λ·corr·∂corr/∂μ，而 ∂corr/∂μ 是 R_θ 的雅可比，优化器可把它往任意方向调，
        # 把 μ 推成反相小场（corr −0.5 塌缩的根因）。detach 后 ∂corr/∂μ=0，μ 的梯度只剩
        # x̂_final=μ+corr 里那条 ∂/∂μ=1 的干净路径，ridge 均衡 corr=(X−μ)/(1+λ) 才成立。
        diff = (flat(x_hat) - flat(mu).detach()) / HU_SCALE
        mu_k = flat(mu).detach() / HU_SCALE
        trust_k = flat(trust)
        d_n = flat(d_j_n)
        inp = torch.cat([diff, mu_k, trust_k, d_n], dim=1)   # [B*Z, 4, H, W]
        corr = self.net(inp) * HU_SCALE                     # [B*Z, 1, H, W] → HU
        corr = corr.reshape(B, 1, Z, H, W)
        if metal_mask is not None:
            corr = corr * metal_mask                        # 只在金属区修正，非金属区 corr=0
        return mu + corr
