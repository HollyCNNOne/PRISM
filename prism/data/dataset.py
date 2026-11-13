"""data —— Theta 序列 Dataset（Phase 4）。

数据契约（一个样本 = 一个体 = 某 patient × dose × geometry）：
    thetas:     [Z] 个逐片物理量 dict，键 {x_M0, d_j, g} + 可选 {tau_b, w_b}，
               各值 [1, H, W]（分带时 tau_b/w_b 为 [nb, H, W]）。
    X_clean:    [Z, 1, H, W] 干净组织（真值，无金属伪影），Q 监督 + 下游损失用。
    x_hat:      [Z, 1, H, W] Baseline 带伪影观测图（下游 R_θ 的观测输入）。
    metal_mask: [Z, 1, H, W] 金属掩膜（1=金属，0=组织）；损失遮盖取其补 I = 1−metal_mask。
    meta:       {dose, geometry, patient}，三轴字段。

collate_theta 把多体样本堆成 batch：
    thetas → [Z] dict，各键 [B, ...]；X_clean/metal_mask/x_hat → [B, Z, 1, H, W]；meta → list。

实际 IO（.npy/.pt 读取、AGIR 产物解析）在 loader 回调里，训练脚本注入；本模块只定契约
与 collate。
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.utils.data import Dataset


class ThetaDataset(Dataset):
    """按样本清单加载的 Theta 序列 Dataset。

    samples: list；每个元素要么是已组好的 sample dict，要么是 spec（由 loader 转成 sample）。
    loader: callable(spec) -> sample dict；None 时 samples 已是 sample dict。
    """

    def __init__(self, samples, loader: Optional[Callable] = None):
        self.samples = samples
        self.loader = loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return self.loader(s) if self.loader is not None else s


def collate_theta(batch):
    """batch: list[sample] → batched dict。

    thetas: [Z] dict（各键堆成 [B, ...]）；X_clean/metal_mask: [B, Z, 1, H, W]；meta: list。
    """
    Z = len(batch[0]["thetas"])
    keys = list(batch[0]["thetas"][0].keys())
    thetas = [
        {k: torch.stack([b["thetas"][z][k] for b in batch], dim=0) for k in keys}
        for z in range(Z)
    ]
    X_clean = torch.stack([b["X_clean"] for b in batch], dim=0)      # [B, Z, 1, H, W]
    metal_mask = torch.stack([b["metal_mask"] for b in batch], dim=0)
    x_hat = torch.stack([b["x_hat"] for b in batch], dim=0)          # [B, Z, 1, H, W]
    meta = [b["meta"] for b in batch]
    out = {"thetas": thetas, "X_clean": X_clean, "metal_mask": metal_mask,
           "x_hat": x_hat, "meta": meta}
    if "li" in batch[0]:
        out["li"] = torch.stack([b["li"] for b in batch], dim=0)     # [B, Z, 1, H, W]
    return out


def load_slice_phys(sample_id: str, phys_dir, target_dir, baseline_dir,
                    li_dir, mask_dir, h: int = 512, w: int = 512):
    """Read one 3D_Coil sample's physical quantities + images → per-slice dict.

    Shape contract (corrected — d_j is a SCALAR, not banded):
        X      = Target   clean CT        [1, H, W]  float32 HU
        x_hat  = Baseline artifact        [1, H, W]  float32 HU
        x_M0   = metal prior              [1, H, W]  (mask·metal_mu)
        g      = gate                     [1, H, W]
        d_j    = data sufficiency         [1, H, W]  (SCALAR, band-independent)
        tau_b  = radial band trust        [4, H, W]  (0-placeholder until LFR wired)
        w_b    = radial band share        [4, H, W]  (0-placeholder until LFR wired)
        mask   = metal volume fraction    [1, H, W]  (1=metal)

    The 4-band structure lives in tau_b/w_b (radial LFR, via
    radial_lfr_from_params → radial_lfr_to_weights), NOT in d_j — d_j is a
    scalar AᵀW(A·1) with no radial band dimension. Calling
    radial_lfr_to_weights(d_j) would be a category error: that function takes
    lfr_radial [nb,...], not d_j.
    """
    import os

    import numpy as np

    def _img(path):
        return torch.from_numpy(
            np.fromfile(path, dtype=np.float32).reshape(1, h, w))

    def _np(path, shape):
        arr = np.load(path).astype(np.float32)
        if tuple(arr.shape) != tuple(shape):
            raise ValueError(f"{path}: expected {tuple(shape)}, got {arr.shape}")
        return torch.from_numpy(arr)

    return {
        "X": _img(os.path.join(target_dir, f"{sample_id}.raw")),
        "x_hat": _img(os.path.join(baseline_dir, f"{sample_id}.raw")),
        "li": _img(os.path.join(li_dir, f"{sample_id}.raw")),
        "x_M0": _np(os.path.join(phys_dir, f"{sample_id}_x_M0.npy"), (1, h, w)),
        "g": _np(os.path.join(phys_dir, f"{sample_id}_g.npy"), (1, h, w)),
        "d_j": _np(os.path.join(phys_dir, f"{sample_id}_d_j.npy"), (1, h, w)),
        "tau_b": _np(os.path.join(phys_dir, f"{sample_id}_tau_b.npy"), (4, h, w)),
        "w_b": _np(os.path.join(phys_dir, f"{sample_id}_w_b.npy"), (4, h, w)),
        "mask": _img(os.path.join(mask_dir, f"{sample_id}.raw")),
    }


def collate_slices(batch):
    """batch: list[per-slice dict] → batched dict（各键按 dim=0 堆成 [B, ...]）。

    与 collate_theta（序列视图）互补：本函数用于「平铺单切片」视图，各键独立堆叠，
    [1,H,W] → [B,1,H,W]、[4,H,W] → [B,4,H,W]，dtype 不变。
    """
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def make_theta_sequence_loader(target_dir, baseline_dir, li_dir, mask_dir,
                               phys_dir, poison=None, h: int = 512, w: int = 512):
    """Return a `loader(spec) -> sample dict` for ThetaDataset, wiring in the
    false-positive injection (poison) BEFORE collate_theta.

    一个 spec = 一次植入的连续切片 sample_id 列表（共享同一 sequence_id），loader 逐片
    读物理量 + 图像 → 组出 thetas [Z] / X_clean [Z,1,H,W] / metal_mask [Z,1,H,W]，
    再（可选）把底图里的真牙（HU>teeth_hu）投毒进 x_M0 通道，最后返回 sample dict。

    poison: None（不投毒）| dict（作为 inject_false_metal 的 kwargs，如
        {'prob': 0.5, 'frac': 1.0, 'teeth_hu': 2500.0, 'metal_mu': None}）。
        metal_mu=None → 自动取本序列 x_M0 峰值（假金属与真线圈同幅值）。

    用法（训练脚本）：
        ds = ThetaDataset(specs, loader=make_theta_sequence_loader(
            target_dir, baseline_dir, li_dir, mask_dir, phys_dir,
            poison={'prob': 0.5}))
        dl = DataLoader(ds, batch_size=B, collate_fn=collate_theta)

    投毒只改 x_M0（SSM 输入先验），不动 metal_mask（损失遮盖 I=1−metal_mask 仍指向
    真线圈；牙齿是真组织，必须被监督）。
    """
    from .poison import inject_false_metal

    def loader(spec):
        slices = [load_slice_phys(sid, phys_dir, target_dir, baseline_dir,
                                  li_dir, mask_dir, h, w) for sid in spec]
        Z = len(slices)
        # 分带量 tau_b/w_b 目前由 3D_Coil 写成 0 占位（径向 LFR 未派生）。全零时必须走
        # 标量退化（τ=w=1），否则 zero 会把观测项 d_j/d_ref 乘成 0（杀死观测）。以首片
        # 为准判断本序列是否已派生分带量（同一数据集内一致）。
        banded = bool((slices[0]["tau_b"] != 0).any())
        thetas = []
        for s in slices:
            t = {"x_M0": s["x_M0"], "d_j": s["d_j"], "g": s["g"]}
            if banded:
                t["tau_b"] = s["tau_b"]
                t["w_b"] = s["w_b"]
            thetas.append(t)
        X_clean = torch.stack([s["X"] for s in slices], dim=0)      # [Z,1,H,W]
        metal_mask = torch.stack([s["mask"] for s in slices], dim=0)  # [Z,1,H,W]
        x_hat = torch.stack([s["x_hat"] for s in slices], dim=0)     # [Z,1,H,W] Baseline 观测
        li = torch.stack([s["li"] for s in slices], dim=0)          # [Z,1,H,W] 经典 LI-MAR（A2 用）

        if poison is not None:
            # 投毒在整段 z 上做（z 连续），逐片写回 x_M0。
            import numpy as np
            x_M0 = torch.stack([t["x_M0"] for t in thetas], dim=0)  # [Z,1,H,W]
            x_M0 = torch.from_numpy(inject_false_metal(
                x_M0.numpy(), X_clean.numpy(), **poison))
            for z, t in enumerate(thetas):
                t["x_M0"] = x_M0[z]

        return {"thetas": thetas, "X_clean": X_clean, "x_hat": x_hat,
                "metal_mask": metal_mask, "li": li, "meta": {"sequence": spec}}

    return loader
