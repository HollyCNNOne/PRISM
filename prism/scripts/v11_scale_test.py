#!/usr/bin/env python
"""v11_scale_test.py —— v1.1-scale-fix 控制变量验证（10 步 Overfit）。

对照官方 FIND-Net 的三处契约做微创修复：
  1. 输入动态范围：HU → (x+1500)/5000 → clip[0,1] → ×255 → [0,255]（find_net.forward）
  2. 输出反归一化：X/255 → ×5000 −1500 → HU（find_net.reconstruct）
  3. Mask 极性：find_net.forward 内部翻成非金属（官方 non_Mask），传入 1=金属（数据契约）即可
（GroupNorm 仍保留、不删 ReLU——那两项留待 v1.1+。）

判据：组织区 RMSE 是否跌破 baseline、corr 是否转正、raw X max 是否落在 [0,255] 域、
末步负像素数 >0（证明能画空气）。末步出 4 列图 Target|Baseline|Prediction|Error。

用法（服务器 pzneuro env）：
  python PRISM/scripts/v11_scale_test.py
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
OFF, SCALE = 1500.0, 5000.0


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
    mask_t = torch.from_numpy(msk)[:, None].float().to(dev)   # 1=金属（find_net 内部翻成非金属）
    tgt_t = torch.from_numpy(tgt)[:, None].float().to(dev)

    fn = FINDNet(FINDNetArgs()).to(dev).train()
    opt = torch.optim.Adam(fn.parameters(), lr=2e-4)

    # Z=1 形状喂给 downstream_reconstruction_loss（loss 仍吃 1=金属，内部 1-metal）
    X_clean = tgt_t.unsqueeze(2)
    metal = mask_t.unsqueeze(2)

    tissues = [(tgt[i] > -500) & (msk[i] < 0.5) for i in range(2)]
    b_rmse = [float(np.sqrt(np.mean((base[i][tissues[i]] - tgt[i][tissues[i]]) ** 2)))
              for i in range(2)]
    b_corr = [float(np.corrcoef(base[i][tissues[i]], tgt[i][tissues[i]])[0, 1])
              for i in range(2)]
    print(f"[v11-scale] device={dev} slices={SIDS}", flush=True)
    print(f"baseline: tissue_RMSE={b_rmse[0]:.0f}/{b_rmse[1]:.0f}  "
          f"corr={b_corr[0]:.4f}/{b_corr[1]:.4f}", flush=True)

    for step in range(10):
        opt.zero_grad()
        pred = fn.reconstruct(ct_ma, lict, mask_t)   # [B,1,H,W] HU（内部入×255/出÷255）
        x_hat_z = pred.unsqueeze(2)
        loss = downstream_reconstruction_loss(x_hat_z, X_clean, metal)
        loss.backward()
        opt.step()
        if (step + 1) in (1, 3, 5, 10):
            with torch.no_grad():
                _, ListX, _ = fn.forward(ct_ma, lict, mask_t)
                rawX = ListX[-1]                                   # [0,255] 域
                hu = (rawX / 255.0 * SCALE - OFF).cpu().numpy()[:, 0]
            rmse = [float(np.sqrt(np.mean((hu[i][tissues[i]] - tgt[i][tissues[i]]) ** 2)))
                    for i in range(2)]
            corr = [float(np.corrcoef(hu[i][tissues[i]], tgt[i][tissues[i]])[0, 1])
                    for i in range(2)]
            print(f"[step {step+1:02d}/10] loss={loss.item():8.1f} "
                  f"rawXmax={rawX.max().item():7.1f} HUmin/max={hu[0].min():6.0f}/{hu[0].max():6.0f} "
                  f"tissue_RMSE={rmse[0]:5.0f}/{rmse[1]:5.0f} corr={corr[0]:+.3f}/{corr[1]:+.3f}",
                  flush=True)

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
    fig.suptitle(f"v1.1-scale-fix overfit 10-step  {SIDS[0]}\n"
                 f"pred min/max/mean={p0.min():.1f}/{p0.max():.1f}/{p0.mean():.1f} | "
                 f"tgt min/max/mean={tgt0.min():.1f}/{tgt0.max():.1f}/{tgt0.mean():.1f}",
                 fontsize=11)
    out = "/root/autodl-tmp/PRISM/training_data_astra/viz/v11_scale_fix.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"[save] {out}", flush=True)
    print(f"[check] pred neg-pixel count={(p0 < 0).sum()} (应 >0，证明能画空气)", flush=True)


if __name__ == "__main__":
    main()
