#!/usr/bin/env python
"""v11_verify.py —— v1.1 修复后验证：clamp(0,1) + 梯度裁剪 是否解除爆炸并唤醒组织学习。

200 步 overfit（2 片），在 [1,10,25,50,100,200] 打：
- loss、raw X max（应 ≤1.0，验证钳制生效）
- denorm HU min/max（应 ∈[-1500,3500]，验证爆炸解除）
- 组织区 RMSE/corr（vs baseline，验证组织是否长出来）
末步出 4 列图 + 负像素数。
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/prism_framework")

from prism.models.find_net import FINDNet, FINDNetArgs  # noqa: E402
from prism.training.losses import downstream_reconstruction_loss  # noqa: E402

D = "/root/autodl-tmp/PRISM/training_data_astra/train"
SIDS = ["sample_000278", "sample_000825"]
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
    mask_t = torch.from_numpy(msk)[:, None].float().to(dev)
    tgt_t = torch.from_numpy(tgt)[:, None].float().to(dev)

    fn = FINDNet(FINDNetArgs()).to(dev).train()
    opt = torch.optim.Adam(fn.parameters(), lr=2e-4)
    X_clean = tgt_t.unsqueeze(2)
    metal = mask_t.unsqueeze(2)
    tissues = [(tgt[i] > -500) & (msk[i] < 0.5) for i in range(2)]

    def stats():
        with torch.no_grad():
            X0, ListX, _ = fn(ct_ma, lict, mask_t)
            X = ListX[-1]
            hu = (X * SCALE - OFF).cpu().numpy()[:, 0]
        rmse = [float(np.sqrt(np.mean((hu[i][tissues[i]] - tgt[i][tissues[i]]) ** 2)))
                for i in range(2)]
        corr = [float(np.corrcoef(hu[i][tissues[i]], tgt[i][tissues[i]])[0, 1])
                for i in range(2)]
        return X.max().item(), float(hu.min()), float(hu.max()), rmse, corr

    b_rmse = [float(np.sqrt(np.mean((base[i][tissues[i]] - tgt[i][tissues[i]]) ** 2)))
              for i in range(2)]
    b_corr = [float(np.corrcoef(base[i][tissues[i]], tgt[i][tissues[i]])[0, 1])
              for i in range(2)]
    print(f"baseline: tissue_RMSE={b_rmse[0]:.0f}/{b_rmse[1]:.0f}  "
          f"corr={b_corr[0]:.4f}/{b_corr[1]:.4f}", flush=True)
    print(f"{'step':>5} {'loss':>8} {'rawXmax':>8} {'HU_min':>8} {'HU_max':>8} "
          f"{'RMSE0':>7} {'RMSE1':>7} {'corr0':>6} {'corr1':>6}", flush=True)

    for step in range(200):
        opt.zero_grad()
        X0, ListX, ListA = fn(ct_ma, lict, mask_t)
        X = ListX[-1]
        x_hat_z = X * SCALE - OFF
        loss = downstream_reconstruction_loss(x_hat_z.unsqueeze(2), X_clean, metal)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fn.parameters(), max_norm=1.0)
        opt.step()
        if (step + 1) in (1, 10, 25, 50, 100, 200):
            rxmax, hmin, hmax, rmse, corr = stats()
            print(f"{step+1:5d} {loss.item():8.1f} {rxmax:8.3f} {hmin:8.0f} {hmax:8.0f} "
                  f"{rmse[0]:7.0f} {rmse[1]:7.0f} {corr[0]:6.3f} {corr[1]:6.3f}", flush=True)

    with torch.no_grad():
        X0, ListX, _ = fn(ct_ma, lict, mask_t)
        pred = (ListX[-1] * SCALE - OFF).cpu().numpy()[:, 0]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p0 = pred[0]
    arrs = [tgt[0], base[0], p0, np.abs(p0 - tgt[0])]
    titles = ["Target", "Baseline", "Prediction", "Error"]
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.5))
    for ax, a, t in zip(axes, arrs, titles):
        vmin, vmax, cmap = (-150, 350, "gray") if t != "Error" else (0, 350, "hot")
        ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(t)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"v1.1 verify 200-step {SIDS[0]}\n"
                 f"pred min/max/mean={p0.min():.0f}/{p0.max():.0f}/{p0.mean():.0f} | "
                 f"tgt={tgt[0].min():.0f}/{tgt[0].max():.0f}/{tgt[0].mean():.0f}")
    out = "/root/autodl-tmp/PRISM/training_data_astra/viz/v11_verify.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"[save] {out}", flush=True)
    print(f"[check] neg-pixel count={(p0 < 0).sum()} (应>0)", flush=True)


if __name__ == "__main__":
    main()
