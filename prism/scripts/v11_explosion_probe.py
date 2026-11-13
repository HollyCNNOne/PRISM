#!/usr/bin/env python
"""v11_explosion_probe.py —— 定位 v1.1 数值爆炸根因。

不加 clamp，只加梯度裁剪 + 逐步打印：
- loss、clip 前 grad_norm
- 原始输出（反归一化前，理论上应∈[0,1]）min/mean/max
- 反归一化后 HU min/max
最后定位爆炸体素（HU>3500）落在金属区 / 空气区 / 组织区，确认是不是「掩膜外区域无监督 + ReLU 上界无界」。
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

    print(f"[probe] device={dev} lr=2e-4 clip=max_norm=1.0", flush=True)
    for step in range(10):
        opt.zero_grad()
        X0, ListX, ListA = fn(ct_ma, lict, mask_t)   # 内部已归一化输入，X 仍在 [0,?] 域
        X = ListX[-1]                                 # 原始输出（反归一化前）
        x_hat_z = X * SCALE - OFF                     # 与 reconstruct 一致的反归一化
        loss = downstream_reconstruction_loss(x_hat_z.unsqueeze(2), X_clean, metal)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(fn.parameters(), max_norm=1.0)
        opt.step()
        with torch.no_grad():
            xm, xM, xmean = X.min().item(), X.max().item(), X.mean().item()
            hm, hM = x_hat_z.min().item(), x_hat_z.max().item()
        print(f"[step {step+1:02d}] loss={loss.item():8.1f} grad_norm={gn:6.2f} | "
              f"raw X min/mean/max={xm:+.3f}/{xmean:+.3f}/{xM:+.3f} | "
              f"denorm HU min/max={hm:+.1f}/{hM:+.1f}", flush=True)

    # 定位爆炸体素（片 0）
    with torch.no_grad():
        _, ListX, _ = fn(ct_ma, lict, mask_t)
        hu = (ListX[-1][0, 0].cpu().numpy()) * SCALE - OFF
    t0 = tgt[0]
    m0 = msk[0]
    metal_region = m0 > 0.5
    air_region = t0 <= -500
    tissue_region = (t0 > -500) & (~metal_region)
    boom = hu > 3500
    print(f"=== 爆炸体素(HU>3500)定位（片 000278）===", flush=True)
    print(f"  总数={int(boom.sum())}  金属区={int(boom[metal_region].sum())}  "
          f"空气区={int(boom[air_region].sum())}  组织区={int(boom[tissue_region].sum())}",
          flush=True)
    am = int(hu.argmax())
    y, x = am // 512, am % 512
    print(f"  HU max={hu.max():.1f} @ ({y},{x})  metal={m0[y, x]:.2f}  tgt={t0[y, x]:.1f}",
          flush=True)


if __name__ == "__main__":
    main()
