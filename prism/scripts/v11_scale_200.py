#!/usr/bin/env python
"""v11_scale_200.py —— v1.1-scale-fix 200 步最终验收（同一单 Batch，控制变量）。

保留 v1.1 三处契约（入 ×255 / 出 ÷255 / Mask 由 find_net 内部翻成非金属），不删 ReLU（v1.1+ 暂缓）。
200 步 overfit（large/giant 两片，同 seed），在 [1,10,25,50,100,200] 打：
loss、raw X max（[0,255] 域）、HU min/max（无爆炸）、组织区 RMSE/corr（vs baseline）。
末步出 2×4 图（两片各 Target|Baseline|Prediction|Error，窗 [-150,350]，Error 热图）。

验收标准：Prediction 星芒显著削弱、向 Target 靠拢；Error 热图变暗；Loss 正常下降。
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
SIDS = ["sample_000278", "sample_000825"]  # large / giant
OFF, SCALE = 1500.0, 5000.0
STEPS = 200


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

    X_clean = tgt_t.unsqueeze(2)
    metal = mask_t.unsqueeze(2)

    tissues = [(tgt[i] > -500) & (msk[i] < 0.5) for i in range(2)]
    b_rmse = [float(np.sqrt(np.mean((base[i][tissues[i]] - tgt[i][tissues[i]]) ** 2)))
              for i in range(2)]
    b_corr = [float(np.corrcoef(base[i][tissues[i]], tgt[i][tissues[i]])[0, 1])
              for i in range(2)]
    print(f"[v11-scale-200] device={dev} slices={SIDS} steps={STEPS}", flush=True)
    print(f"baseline: tissue_RMSE={b_rmse[0]:.0f}/{b_rmse[1]:.0f}  "
          f"corr={b_corr[0]:.4f}/{b_corr[1]:.4f}", flush=True)
    print(f"{'step':>5} {'loss':>8} {'rawXmax':>8} {'HUmin':>7} {'HUmax':>7} "
          f"{'RMSE0':>7} {'RMSE1':>7} {'corr0':>7} {'corr1':>7}", flush=True)

    for step in range(STEPS):
        opt.zero_grad()
        pred = fn.reconstruct(ct_ma, lict, mask_t)   # HU（内部入×255/出÷255）
        x_hat_z = pred.unsqueeze(2)
        loss = downstream_reconstruction_loss(x_hat_z, X_clean, metal)
        loss.backward()
        opt.step()
        if (step + 1) in (1, 10, 25, 50, 100, 200):
            with torch.no_grad():
                _, ListX, _ = fn.forward(ct_ma, lict, mask_t)
                rawX = ListX[-1]
                hu = (rawX / 255.0 * SCALE - OFF).cpu().numpy()[:, 0]
            rmse = [float(np.sqrt(np.mean((hu[i][tissues[i]] - tgt[i][tissues[i]]) ** 2)))
                    for i in range(2)]
            corr = [float(np.corrcoef(hu[i][tissues[i]], tgt[i][tissues[i]])[0, 1])
                    for i in range(2)]
            print(f"{step+1:5d} {loss.item():8.1f} {rawX.max().item():8.1f} "
                  f"{hu[0].min():7.0f} {hu[0].max():7.0f} {rmse[0]:7.0f} {rmse[1]:7.0f} "
                  f"{corr[0]:7.3f} {corr[1]:7.3f}", flush=True)

    with torch.no_grad():
        pred = fn.reconstruct(ct_ma, lict, mask_t).cpu().numpy()[:, 0]

    fig, axes = plt.subplots(2, 4, figsize=(21, 11))
    for r in range(2):
        tgt0, base0, p0 = tgt[r], base[r], pred[r]
        arrs = [tgt0, base0, p0, np.abs(p0 - tgt0)]
        titles = ["Target", "Baseline", "Prediction", "Error |pred-tgt|"]
        for c, (a, t) in enumerate(zip(arrs, titles)):
            vmin, vmax, cmap = (-150, 350, "gray") if "Error" not in t else (0, 350, "hot")
            im = axes[r, c].imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            if r == 0:
                axes[r, c].set_title(t, fontsize=12)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(f"{SIDS[r]}\npred mean={p0.mean():.0f}", fontsize=10)
    fig.suptitle(f"v1.1-scale-fix  200-step overfit\n"
                 f"slice0 pred min/max={pred[0].min():.0f}/{pred[0].max():.0f} | "
                 f"tgt={tgt[0].min():.0f}/{tgt[0].max():.0f}",
                 fontsize=11)
    out = "/root/autodl-tmp/PRISM/training_data_astra/viz/v11_scale_200.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"[save] {out}", flush=True)
    for r in range(2):
        print(f"[check] slice{r} neg-pixel={(pred[r] < 0).sum()} (应>0)", flush=True)


if __name__ == "__main__":
    main()
