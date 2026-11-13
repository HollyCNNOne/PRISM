"""augment —— Stage 1/2 二维单切片强增强（纯 torch，无 torchvision 依赖）。

动机：1386 片样本极度稀缺，FIND-Net 容易死记硬背。这里只做二维几何/遮挡增强，
不破坏 z 连续性（Stage 1 练单兵能力；Stage 2 的 FIND-Net-only 分支同用）。

四张图（CT_ma / LIct / Mask / Target）必须共用同一个采样 grid，保证空间对齐：
  - 旋转 ±15°（affine_grid）
  - 弹性形变（粗网格随机位移场，双线性上采样，σ 控制像素幅度）
  - Cutout 只挖输入（CT_ma / LIct），Target / Mask 保持完整（盲区要由网络从上下文+先验填）

Mask 是金属体积分数场（非二值，[0,~0.15]），bilinear 采样后仍是有意义的小数，直接共用。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def elastic_disp(B, H, W, device, sigma=8.0, grid_div=16):
    """随机平滑位移场，返回**归一化 grid 单位**（可直接加到 affine grid 上）。

    粗网格 random + 双线性上采样；sigma 是位移幅度（像素），换算成归一化单位：
    align_corners=False 时 d(normalized)/d(pixel) = 2/size。
    """
    gh = max(H // grid_div, 2)
    gw = max(W // grid_div, 2)
    d = torch.randn(B, 2, gh, gw, device=device)
    d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)  # [B,2,H,W]
    d[:, 0] = d[:, 0] * (2.0 * sigma / W)   # x 方向（像素 → 归一化）
    d[:, 1] = d[:, 1] * (2.0 * sigma / H)   # y 方向
    return d.permute(0, 2, 3, 1)            # [B,H,W,2]（grid_sample 通道在后）


def make_aug_grid(B, H, W, device, angle_deg=0.0, disp=None):
    """一张 grid 同时承载旋转 + 弹性形变，四通道复用保证对齐。

    angle_deg: 旋转角（度，±15）。disp: 归一化位移场（elastic_disp 产出，None=无弹性）。
    """
    theta = torch.zeros(B, 2, 3, device=device)
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    theta[:, 0, 0] = c
    theta[:, 0, 1] = -s
    theta[:, 1, 0] = s
    theta[:, 1, 1] = c
    grid = F.affine_grid(theta, torch.Size([B, 1, H, W]), align_corners=False)
    if disp is not None:
        grid = grid + disp
    return grid


def warp(tensor, grid):
    return F.grid_sample(tensor, grid, align_corners=False, mode="bilinear",
                         padding_mode="border")


def cutout(tensor, n_holes=1, max_size=64, fill=0.0):
    """随机矩形挖洞（遮挡输入通道，逼网络从上下文 + 先验填盲区）。返回同形 tensor。

    洞的位置/大小 batch 内共享（batch 通常 1~2，够用）；只作用于输入，不碰 target/mask。
    """
    B, _, H, W = tensor.shape
    out = tensor.clone()
    for _ in range(n_holes):
        h = int(torch.randint(8, max_size + 1, (1,)).item())
        w = int(torch.randint(8, max_size + 1, (1,)).item())
        y = int(torch.randint(0, max(H - h, 1) + 1, (1,)).item())
        x = int(torch.randint(0, max(W - w, 1) + 1, (1,)).item())
        out[..., y:y + h, x:x + w] = fill
    return out
