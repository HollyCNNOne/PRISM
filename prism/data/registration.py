"""registration —— 2D 面内旋转转正（模块1，Z-BCT 前置，on-the-fly 不落盘）。

华西真实底图 raw_Dataset/Huashi/patients_head1 中 head1/5/6/7/8 是"右旋 45°"
（面内顺时针旋转）。Z-BCT 的 z 轴频域追踪要求解剖柱沿 z 对齐，故在数据进入模型前
对每片做"左旋 45°"转正。

铁律：**不改原始 .raw 文件**——旋转在 DataLoader 加载时 on-the-fly 完成，原件保持。

约定：
  - 2D 面内旋转（逐片独立，z 序与片数不变）。
  - "右旋 45°"（顺时针）→ 转正 = 逆时针；具体 sign（+45 还是 −45）需与真实切片
    肉眼核对一次，`straighten_right_rotated` 默认取 −45，若转完更斜就翻正负号。
  - torch affine_grid + grid_sample，零第三方依赖（角度已知，无需 SimpleITK/ANTs）。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _rotation_matrix_2d(angle_deg: float) -> torch.Tensor:
    """2D 旋转矩阵（逆时针为正），返回 2×3 仿射矩阵。"""
    c = math.cos(math.radians(angle_deg))
    s = math.sin(math.radians(angle_deg))
    return torch.tensor([[c, -s, 0.0],
                         [s, c, 0.0]], dtype=torch.float32)


def rotate_slices(images: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """对 [B, C, H, W]（或 [C, H, W]）逐片面内旋转 angle_deg 度。

    angle_deg > 0 = 逆时针；< 0 = 顺时针。返回同 shape。
    """
    single = images.dim() == 3
    x = images.unsqueeze(0) if single else images
    B, C, H, W = x.shape

    theta = _rotation_matrix_2d(angle_deg).to(x.device).unsqueeze(0).expand(B, 2, 3)
    grid = F.affine_grid(theta, (B, C, H, W), align_corners=False)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros",
                        align_corners=False)

    return out.squeeze(0) if single else out


def straighten_right_rotated(images: torch.Tensor, angle_deg: float = 45.0) -> torch.Tensor:
    """转正"右旋 angle_deg"的切片：逆时针旋转 -angle_deg。

    images: [B, C, H, W] 或 [C, H, W]（右旋后的切片）。
    返回转正后的同 shape 张量。
    """
    return rotate_slices(images, -angle_deg)
