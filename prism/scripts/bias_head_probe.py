#!/usr/bin/env python
"""bias_head_probe.py —— 低频 bias 校正头探针（判据：能否从金属几何预测 carry 偏置）。

背景：方向性信任已判负（残差各向同性 aniso≈1，低频束硬化 shading）。追 +77 carry 偏置的
正确路径 = 学一个「读金属几何(x_M0,d_j)、输出低频光滑场」的 bias 校正头 δ_head，把系统
偏置 δ=x̂−target 从组织里分离（低频约束天然分离：组织是高频，头输出不了）。

探针（先验证、再整合，吸取方向滤波教训）：
  单独训 δ_head：输入 (x_M0, d_j) 2 通道 → 下采样 32×32 瓶颈（硬低频约束）→ 上采样
  δ_pred。监督 δ_pred ≈ δ = x̂−target，只在部分阴影区（carry 相关，非金属）。
  判据（held-out patients_head2 的部分阴影区）：
    R² > ~0.3  → 偏置可从金属几何预测，值得整合进 B8（再跑完整阴影 bias 对比）。
    R² ≈ 0    → 偏置是噪声、不可预测，bias 校正头也判负，+77 接受为闭环正确行为。

为什么只读 x_M0/d_j 不读 x̂：否则会学成恒等（δ_pred→x̂，一减变 0）。只读几何 → 只能输出
「金属几何决定的那部分」= 系统偏置。为什么只监督部分阴影：完全阴影观测被 K=1 丢弃、对 μ
无关，其 δ 巨大（~4000 HU）会主导 MSE 把头带偏；部分阴影（~380 HU）才是 carry 来源。

用法（服务器 pzneuro env）：
    python bias_head_probe.py --data-dir /root/autodl-tmp/PRISM/training_data \
        --epochs 30 --batch-size 8 --device cuda
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_FRAMEWORK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _FRAMEWORK_ROOT)

H = W = 512


# ---------------------------------------------------------------- 数据

def list_meta(meta_dir, holdout_patient):
    """按 split/patient 拆 (train_ids, heldout_ids)。"""
    train, heldout = [], []
    for f in glob.glob(os.path.join(meta_dir, "*.json")):
        m = json.load(open(f))
        if m.get("split", "train") != "train":
            continue
        sid = m["sample_id"]
        if m.get("patient_id") == holdout_patient:
            heldout.append(sid)
        else:
            train.append(sid)
    train.sort()
    heldout.sort()
    return train, heldout


class BiasDataset(Dataset):
    """逐片加载，返回 (inputs [2,H,W], delta [1,H,W], weight [1,H,W])。

    inputs: ch0 = x_M0/35000, ch1 = log1p(d_j/d_ref)/5
    delta : δ = x̂ − target（HU）
    weight: 部分阴影(1/9≤r<9, 非金属)=1.0，其余=0（完全阴影/充分区/金属不监督）
    """

    def __init__(self, ids, target_dir, baseline_dir, phys_dir, d_ref):
        self.ids = ids
        self.target_dir = target_dir
        self.baseline_dir = baseline_dir
        self.phys_dir = phys_dir
        self.d_ref = d_ref

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        x_hat = np.fromfile(os.path.join(self.baseline_dir, f"{sid}.raw"),
                            dtype=np.float32).reshape(H, W)
        target = np.fromfile(os.path.join(self.target_dir, f"{sid}.raw"),
                             dtype=np.float32).reshape(H, W)
        x_M0 = np.load(os.path.join(self.phys_dir, f"{sid}_x_M0.npy")).astype(np.float32)[0]
        d_j = np.load(os.path.join(self.phys_dir, f"{sid}_d_j.npy")).astype(np.float32)[0]

        r = d_j / self.d_ref
        metal = x_M0 > 0
        partial = (r >= 1.0 / 9.0) & (r < 9.0) & (~metal)

        ch0 = (x_M0 / 35000.0).astype(np.float32)
        ch1 = (np.log1p(r) / 5.0).astype(np.float32)
        inputs = np.stack([ch0, ch1], axis=0)          # [2,H,W]
        delta = (x_hat - target).astype(np.float32)     # [H,W]
        weight = partial.astype(np.float32)             # [H,W]

        return torch.from_numpy(inputs), torch.from_numpy(delta).unsqueeze(0), \
            torch.from_numpy(weight).unsqueeze(0)


# ---------------------------------------------------------------- 网络

class BiasHead(nn.Module):
    """编解码瓶颈网：512→32→512，瓶颈 32×32 硬性约束输出为低频光滑场。"""

    def __init__(self, in_ch=2, base=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, 2, 1), nn.ReLU(),       # 256
            nn.Conv2d(base, base * 2, 3, 2, 1), nn.ReLU(),    # 128
            nn.Conv2d(base * 2, base * 4, 3, 2, 1), nn.ReLU(),  # 64
            nn.Conv2d(base * 4, base * 8, 3, 2, 1), nn.ReLU(),  # 32
        )
        self.bot = nn.Sequential(
            nn.Conv2d(base * 8, base * 8, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(base * 8, base * 8, 3, 1, 1), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1), nn.ReLU(),  # 64
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1), nn.ReLU(),  # 128
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1), nn.ReLU(),      # 256
            nn.ConvTranspose2d(base, 1, 4, 2, 1),                          # 512
        )

    def forward(self, x):
        return self.dec(self.bot(self.enc(x)))


# ---------------------------------------------------------------- 指标

def r2(delta, pred, sel):
    d = delta[sel]
    p = pred[sel]
    ss_res = float(torch.sum((d - p) ** 2))
    ss_tot = float(torch.sum((d - d.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def rmse(delta, pred, sel):
    return float(torch.sqrt(torch.mean((delta[sel] - pred[sel]) ** 2)))


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data")
    ap.add_argument("--holdout-patient", default="patients_head2")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_dir = os.path.join(args.data_dir, "train")
    meta_dir = os.path.join(args.data_dir, "meta")
    target_dir = os.path.join(train_dir, "Target")
    baseline_dir = os.path.join(train_dir, "Baseline")
    phys_dir = os.path.join(train_dir, "Phys")

    d_ref = float(np.load(os.path.join(args.data_dir, f"d_ref_ho_{args.holdout_patient}.npy"))[0])
    print(f"[d_ref] {d_ref:.6g}", flush=True)

    train_ids, held_ids = list_meta(meta_dir, args.holdout_patient)
    print(f"[data] train={len(train_ids)} heldout({args.holdout_patient})={len(held_ids)}",
          flush=True)

    train_ds = BiasDataset(train_ids, target_dir, baseline_dir, phys_dir, d_ref)
    held_ds = BiasDataset(held_ids, target_dir, baseline_dir, phys_dir, d_ref)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    held_dl = DataLoader(held_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    net = BiasHead(in_ch=2, base=args.base).to(args.device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[model] BiasHead(base={args.base}) params={n_params}", flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for ep in range(args.epochs):
        t0 = time.time()
        net.train()
        tot = 0.0
        n_batch = 0
        for inputs, delta, weight in train_dl:
            inputs = inputs.to(args.device)
            delta = delta.to(args.device)
            weight = weight.to(args.device)
            pred = net(inputs)
            # 加权 MSE：只监督部分阴影区（weight 已经 0/1）
            n_w = weight.sum() + 1e-8
            loss = torch.sum(weight * (pred - delta) ** 2) / n_w
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            n_batch += 1
        sched.step()

        # 每个 epoch 后 eval held-out 部分阴影 R² + rmse
        net.eval()
        ss_res = ss_tot = 0.0
        n_vox = 0
        rmse_before = rmse_after = 0.0
        with torch.no_grad():
            for inputs, delta, weight in held_dl:
                inputs = inputs.to(args.device)
                delta = delta.to(args.device)
                weight = weight.to(args.device)
                pred = net(inputs)
                sel = weight > 0.5
                if sel.sum() == 0:
                    continue
                d = delta[sel]
                p = pred[sel]
                ss_res += float(torch.sum((d - p) ** 2))
                ss_tot += float(torch.sum((d - d.mean()) ** 2))
                rmse_before += float(torch.sum(d ** 2))
                rmse_after += float(torch.sum((d - p) ** 2))
                n_vox += int(sel.sum())
        r2_ho = 1.0 - ss_res / max(ss_tot, 1e-12)
        rmse_b = np.sqrt(rmse_before / max(n_vox, 1))
        rmse_a = np.sqrt(rmse_after / max(n_vox, 1))
        print(f"[ep {ep+1}/{args.epochs}] loss={tot/n_batch:.2f} "
              f"heldout_partial R2={r2_ho:.3f} rmse {rmse_b:.0f}->{rmse_a:.0f} "
              f"({time.time()-t0:.1f}s)", flush=True)

    out = os.path.join(args.data_dir, "bias_head_probe.pt")
    torch.save({"model": net.state_dict(), "d_ref": d_ref, "base": args.base}, out)
    print(f"[save] {out}", flush=True)


if __name__ == "__main__":
    main()
