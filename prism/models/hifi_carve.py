"""hifi_carve —— HiFi 式「频域分家 + 内容门控」刻画块（纯 PyTorch，无 mamba CUDA kernel）。

移植 HiFi-Mamba 的 WLBlock（拉普拉斯金字塔分家）+ SimpleGate（内容门控）的精髓：
    low  = 二项式 5 抽头低通 → stride2 下采样 → bilinear 上采样   （低频结构）
    high = x − low                                                （高频残差/细节）
    gate = sigmoid(conv(x))                                        （内容门控：学「哪里该锐化」）
    out  = x + gate ⊙ conv(high)                                   （残差 + 只在有结构处锐化高频）

设计动机：把「高频细节」从特征里显式拆出来单独精修，并让锐化是内容相关的——
有组织结构处锐、平坦/伪影残留处不锐。对照 v1.4 的失败（梯度损失 = 无门控无脑锐化 → 空雕）。

这是「先快速跑一下看有没有效果」的最小注入，不是完整 SS2D（选择性扫描需 CUDA kernel，
服务器无 mamba_ssm/causal_conv1d，故用深度可分离卷积 + 内容门控近似「选择性」）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _binomial_1d_5() -> torch.Tensor:
    # [1,4,6,4,1]/16 二项式 5 抽头（对齐 HiFi binomial_1d_5）
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float32)
    return k / k.sum()


def _gn(channels: int) -> nn.Module:
    """GroupNorm：组数取 ≤8 的最大约数（复用 FIND-Net 的 GroupNorm 约定，batch=1 干净）。"""
    if channels <= 0:
        return nn.Identity()
    g = min(8, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


class HiFiCarve(nn.Module):
    """HiFi 式高频刻画块。输入/输出 [B, C, H, W]，残差 + 内容门控高频精修。"""

    def __init__(self, dim: int, kernel_size: int = 5):
        super().__init__()
        self.dim = dim
        self.pad = kernel_size // 2
        self.register_buffer("k1d", _binomial_1d_5())   # [5]
        g = dim if dim > 1 else 1                        # 深度可分离分组

        # 内容门控：sigmoid(1×1→GN→ReLU→1×1)，网络学「哪里该锐化」
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            _gn(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=True),
        )
        # 高频精修：深度可分离 3×3（只作用于高频残差 high，不碰低频结构）
        self.refine = nn.Conv2d(dim, dim, 3, padding=1, groups=g, bias=False)

        # 初始「可学」：refine 小随机（非零→gate 从 step0 就有梯度）、gate 半开（bias=0→sigmoid≈0.5）。
        # 旧版（refine=0 + gate bias −3）会把块瘫死：refine(high)=0 → gate 梯度恒 0、refine 梯度被 0.047 压 20 倍。
        nn.init.normal_(self.refine.weight, std=0.01)
        nn.init.zeros_(self.gate[-1].bias)

    def _sep_lowpass(self, x: torch.Tensor) -> torch.Tensor:
        """可分离二项式低通 + 2×2 平均池下采样（HiFi SeparableReduce 的轻量等价）。"""
        k = self.k1d.to(x.dtype)
        C = x.shape[1]
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
        kh = k.view(1, 1, 1, -1).expand(C, -1, -1, -1)   # [C,1,1,5]
        x = F.conv2d(x, kh, groups=C)                     # 水平低通
        kv = k.view(1, 1, -1, 1).expand(C, -1, -1, -1)    # [C,1,5,1]
        x = F.conv2d(x, kv, groups=C)                     # 垂直低通
        return F.avg_pool2d(x, kernel_size=2, stride=2)   # stride2 下采样

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self._sep_lowpass(x)                        # [B,C,H/2,W/2]
        low_up = F.interpolate(low, scale_factor=2, mode="bilinear", align_corners=False)
        high = x - low_up                                 # 高频残差（细节）
        gate = torch.sigmoid(self.gate(x))                # [B,C,H,W] 内容门控 ∈ (0,1)
        return x + gate * self.refine(high)               # 残差 + 只在有结构处锐化
