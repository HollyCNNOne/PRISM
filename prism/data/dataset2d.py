"""dataset2d —— Stage 1 / Stage 2(FIND-Net-only) 的二维单切片 Dataset。

把 3D 序列全部打散成单切片：每个样本 = (CT_ma, LIct, Mask, Target) 四张 [1,H,W]。
强增强（旋转/弹性/Cutout）在 __getitem__ 里随机应用，四图共用同一 grid。

映射（与递归循环、FIND-Net 标准一致）：
    CT_ma  = Baseline 脏图（带伪影观测，FIND-Net 的 CT_ma 输入）
    LIct   = LI（经典 LI-MAR，金属 HU 保持到 32700，FIND-Net 的 LIct 基线输入）
    Mask   = 金属体积分数场（非二值，[0,~0.15]）
    Target = 干净组织真值
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from . import augment


def _read_raw(path, h=512, w=512):
    return torch.from_numpy(np.fromfile(path, dtype=np.float32).reshape(1, h, w))


class Slice2DDataset(Dataset):
    def __init__(self, sample_ids, target_dir, baseline_dir, li_dir, mask_dir,
                 augment=True, h=512, w=512):
        self.sample_ids = list(sample_ids)
        self.target_dir = target_dir
        self.baseline_dir = baseline_dir
        self.li_dir = li_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.h, self.w = h, w

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        ct_ma = _read_raw(os.path.join(self.baseline_dir, f"{sid}.raw"), self.h, self.w)
        lict = _read_raw(os.path.join(self.li_dir, f"{sid}.raw"), self.h, self.w)
        mask = _read_raw(os.path.join(self.mask_dir, f"{sid}.raw"), self.h, self.w)
        target = _read_raw(os.path.join(self.target_dir, f"{sid}.raw"), self.h, self.w)

        if self.augment:
            device = ct_ma.device
            angle = float(torch.empty(1).uniform_(-15.0, 15.0).item())
            disp = augment.elastic_disp(1, self.h, self.w, device, sigma=8.0)
            grid = augment.make_aug_grid(1, self.h, self.w, device, angle_deg=angle, disp=disp)
            # 增广函数按 4D [N,C,H,W] 编写；本处 [1,H,W] 临时加 batch 维，增广后 squeeze 回。
            ct_ma = augment.warp(ct_ma[None], grid)[0]
            lict = augment.warp(lict[None], grid)[0]
            mask = augment.warp(mask[None], grid)[0]
            target = augment.warp(target[None], grid)[0]
            # Cutout 只挖输入（CT_ma / LIct），target / mask 保持完整（盲区要由网络填）。
            ct_ma = augment.cutout(ct_ma[None])[0]
            lict = augment.cutout(lict[None])[0]
        return ct_ma, lict, mask, target


def collate2d(batch):
    ct_ma = torch.stack([b[0] for b in batch], dim=0)
    lict = torch.stack([b[1] for b in batch], dim=0)
    mask = torch.stack([b[2] for b in batch], dim=0)
    target = torch.stack([b[3] for b in batch], dim=0)
    return ct_ma, lict, mask, target
