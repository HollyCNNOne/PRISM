#!/usr/bin/env python
"""v11_overfit_eval.py —— v1.1 overfit 的定量组织区评估（decide success/fail）。

与 v11_overfit_test.py 同 seed/同设置，跑 200 步，在 [1,10,25,50,100,200] 打组织区
RMSE/corr 轨迹。组织区 = target>-500 且 metal<0.5（与 B_down 空气+金属屏蔽一致）。
判据：pred 组织区 RMSE 是否跌破 baseline、pred 组织区 corr 是否转正。
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

    tissues = []
    for i in range(2):
        tissues.append((tgt[i] > -500) & (msk[i] < 0.5))

    def stats():
        with torch.no_grad():
            pr = fn.reconstruct(ct_ma, lict, mask_t).cpu().numpy()[:, 0]
        out = []
        for i in range(2):
            tissue = tissues[i]
            t = tgt[i]
            rmse = float(np.sqrt(np.mean((pr[i][tissue] - t[tissue]) ** 2)))
            corr = float(np.corrcoef(pr[i][tissue], t[tissue])[0, 1])
            out.append((rmse, corr))
        return out

    base_rmse = [float(np.sqrt(np.mean((base[i][tissues[i]] - tgt[i][tissues[i]]) ** 2)))
                 for i in range(2)]
    base_corr = [float(np.corrcoef(base[i][tissues[i]], tgt[i][tissues[i]])[0, 1])
                 for i in range(2)]
    print(f"baseline:  tissue_RMSE={base_rmse[0]:.0f}/{base_rmse[1]:.0f}  "
          f"corr={base_corr[0]:.4f}/{base_corr[1]:.4f}", flush=True)

    for step in range(200):
        opt.zero_grad()
        pred = fn.reconstruct(ct_ma, lict, mask_t)
        loss = downstream_reconstruction_loss(pred.unsqueeze(2), X_clean, metal)
        loss.backward()
        opt.step()
        if (step + 1) in (1, 10, 25, 50, 100, 200):
            s = stats()
            print(f"step {step+1:3d}: loss={loss.item():8.1f}  "
                  f"tissue_RMSE={s[0][0]:.0f}/{s[1][0]:.0f}  "
                  f"corr={s[0][1]:.4f}/{s[1][1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
