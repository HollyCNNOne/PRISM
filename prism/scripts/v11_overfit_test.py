#!/usr/bin/env python
"""v11_overfit_test.py —— v1.1 微创手术后的 10 步单 Batch Overfit 测试。

验证目标：标准化翻译服（HU→[0,1] 入 / 反归一化出）+ B_down 空气屏蔽 后，
FIND-Net 是否摆脱 Mode Collapse、能否在 10 步内学会重建组织（长出真实灰脑组织）。

取 2 个带金属最重切片（large/giant，viz_stage1 挑的 streakiest），fresh FIND-Net，
Adam lr=2e-4，10 步，每步打印组织区 masked-MSE（HU_SCALE 归一化，空气+金属全屏蔽），
末步出 4 列图 Target|Baseline|Prediction|Error（窗 [-150,350]）。

用法（服务器 pzneuro env）：
  python PRISM/scripts/v11_overfit_test.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/prism_framework")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prism.models.find_net import FINDNet, FINDNetArgs  # noqa: E402
from prism.training.losses import downstream_reconstruction_loss  # noqa: E402

D = "/root/autodl-tmp/PRISM/training_data_astra/train"
SIDS = ["sample_000278", "sample_000825"]  # large / giant（viz_stage1 挑的 streakiest）


def rd(sid, sub):
    return np.fromfile(os.path.join(D, sub, f"{sid}.raw"), np.float32).reshape(512, 512)


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    base = np.stack([rd(s, "Baseline") for s in SIDS])
    li = np.stack([rd(s, "LI") for s in SIDS])
    msk = np.stack([rd(s, "Mask") for s in SIDS])
    tgt = np.stack([rd(s, "Target") for s in SIDS])

    ct_ma = torch.from_numpy(base)[:, None].float().to(dev)
    lict = torch.from_numpy(li)[:, None].float().to(dev)
    mask_t = torch.from_numpy(msk)[:, None].float().to(dev)
    tgt_t = torch.from_numpy(tgt)[:, None].float().to(dev)

    fn = FINDNet(FINDNetArgs()).to(dev).train()
    opt = torch.optim.Adam(fn.parameters(), lr=2e-4)

    # Z=1 形状喂给 downstream_reconstruction_loss（测真实 B_down 代码路径）
    X_clean = tgt_t.unsqueeze(2)   # [B,1,1,H,W] 当作 [B,Z=1,1,H,W]
    metal = mask_t.unsqueeze(2)

    print(f"[v11] device={dev} batch=2 slices={SIDS}", flush=True)
    for step in range(10):
        opt.zero_grad()
        pred = fn.reconstruct(ct_ma, lict, mask_t)   # [B,1,H,W] 反归一化回 HU
        x_hat_z = pred.unsqueeze(2)                   # [B,1,1,H,W]
        loss = downstream_reconstruction_loss(x_hat_z, X_clean, metal)
        loss.backward()
        opt.step()
        print(f"[step {step+1:02d}/10] loss={loss.item():.4f}", flush=True)

    with torch.no_grad():
        pred = fn.reconstruct(ct_ma, lict, mask_t)
    p0 = pred[0, 0].cpu().numpy()
    tgt0 = tgt[0]

    arrs = [tgt0, base[0], p0, np.abs(p0 - tgt0)]
    titles = ["Target", "Baseline", "Prediction", "Error |pred-tgt|"]
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.5))
    for ax, a, t in zip(axes, arrs, titles):
        vmin, vmax, cmap = (-150, 350, "gray") if "Error" not in t else (0, 350, "hot")
        ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(t, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"v1.1 overfit 10-step  {SIDS[0]}\n"
                 f"pred min/max/mean={p0.min():.1f}/{p0.max():.1f}/{p0.mean():.1f} | "
                 f"tgt min/max/mean={tgt0.min():.1f}/{tgt0.max():.1f}/{tgt0.mean():.1f}",
                 fontsize=11)
    out = "/root/autodl-tmp/PRISM/training_data_astra/viz/v11_overfit.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"[save] {out}", flush=True)
    print(f"[check] pred neg-pixel count={(p0 < 0).sum()} (应 > 0，证明能输出空气/负值)", flush=True)


if __name__ == "__main__":
    main()
