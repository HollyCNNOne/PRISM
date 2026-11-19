#!/usr/bin/env python
"""v14_overfit.py —— v1.4 边缘雕刻 + 高频损失硬化 50 步单序列过拟合验证。

验证 v1.4 三项：λ_ssim 0.1→0.25、新增 Sobel 梯度损失 λ_grad=0.1、组合
Total = masked_MSE + 0.25·SSIM + 0.1·SobelGrad。

warm-start findnet_stage1_v1.1.pt，对 sample_000973 所在序列跑 50 步 A-step，
每步打印 masked MSE / SSIM / SobelGrad 三个分量（从 trainer.last_down_parts 读），
末步出 4 列图（Target|Baseline|Prediction|Error）。禁止直接全量。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/prism_framework")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prism.data.dataset import make_theta_sequence_loader  # noqa: E402
from prism.models.m_network import MNetwork  # noqa: E402
from prism.models.find_net import FINDNet, FINDNetArgs  # noqa: E402
from prism.recursion.algorithm1 import forward_backward  # noqa: E402
from prism.training.trainer import SSMTrainer  # noqa: E402
from prism.utils.config import SSMConfig  # noqa: E402


def _sample_idx(sid):
    return int(sid.rsplit("_", 1)[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", default="sample_000973")
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--resume-findnet",
                    default="/root/autodl-tmp/PRISM/training_data_astra/findnet_stage1_v1.1.pt")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda-ssim", type=float, default=0.25)
    ap.add_argument("--lambda-grad", type=float, default=0.1)
    ap.add_argument("--backfill-r", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/root/autodl-tmp/PRISM/training_data_astra/viz_v12/v14_overfit.png")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    train_dir = os.path.join(args.data_dir, "train")
    meta_dir = os.path.join(args.data_dir, "meta")

    seq_id = None
    for f in glob.glob(os.path.join(meta_dir, "*.json")):
        m = json.load(open(f))
        if m["sample_id"] == args.sid:
            seq_id = m["sequence_id"]
            break
    seq = sorted([json.load(open(f))["sample_id"] for f in glob.glob(os.path.join(meta_dir, "*.json"))
                  if json.load(open(f)).get("sequence_id") == seq_id], key=_sample_idx)
    print(f"[v14] sid={args.sid} seq={len(seq)} slices lambda_ssim={args.lambda_ssim} "
          f"lambda_grad={args.lambda_grad} backfill_r={args.backfill_r}", flush=True)

    d_ref = 4.10023e+06
    loader = make_theta_sequence_loader(
        os.path.join(train_dir, "Target"), os.path.join(train_dir, "Baseline"),
        os.path.join(train_dir, "LI"), os.path.join(train_dir, "Mask"),
        os.path.join(train_dir, "Phys"))

    m_net = MNetwork(nb=1, channels=32, use_velocity=True).to(args.device)
    find_net = FINDNet(FINDNetArgs()).to(args.device)
    if args.resume_findnet and os.path.exists(args.resume_findnet):
        ck = torch.load(args.resume_findnet, map_location=args.device, weights_only=False)
        find_net.load_state_dict(ck["find_net"])
        print(f"[load] find_net from {args.resume_findnet}", flush=True)

    config = SSMConfig(nb=1, channels=32, recursion="bidirectional", d_ref=d_ref,
                       cold_start_init="obs", lambda_q=1.0, lambda_pc=0.1,
                       lambda_down=1.0, lambda_mu=1.0,
                       lambda_ssim=args.lambda_ssim, lambda_grad=args.lambda_grad,
                       backfill_r=args.backfill_r)
    trainer = SSMTrainer(m_net, config, device=args.device, find_net=find_net)
    opt = torch.optim.Adam([
        {"params": m_net.parameters(), "lr": args.lr},
        {"params": find_net.parameters(), "lr": args.lr},
    ], lr=args.lr)
    trainer.optimizer = opt

    sample = loader(seq)
    thetas = [{k: v.unsqueeze(0).to(args.device) for k, v in t.items()} for t in sample["thetas"]]
    X_clean = sample["X_clean"].unsqueeze(0).to(args.device)
    metal_mask = sample["metal_mask"].unsqueeze(0).to(args.device)
    x_hat = sample["x_hat"].unsqueeze(0).to(args.device)
    li = sample["li"].unsqueeze(0).to(args.device)

    print(f"{'step':>4} {'total':>9} {'MSE':>8} {'SSIM':>8} {'Grad':>8}", flush=True)
    for step in range(args.steps):
        total, q, pc, down, mu = trainer.train_step(
            thetas, X_clean, metal_mask, x_hat, clip_grad=0.0, li=li)
        mse, ssim, grad = trainer.last_down_parts
        print(f"{step+1:4d} {total.item():9.3f} {mse.item():8.3f} {ssim.item():8.3f} {grad.item():8.3f}",
              flush=True)

    # 出图
    m_net.eval()
    find_net.eval()
    with torch.no_grad():
        dirty_vol = x_hat.permute(0, 2, 1, 3, 4)
        dirty = [dirty_vol[:, :, z] for z in range(len(thetas))]
        li_vol = li.permute(0, 2, 1, 3, 4)
        li_list = [li_vol[:, :, z] for z in range(len(thetas))]
        mask_input = metal_mask.permute(0, 2, 1, 3, 4)
        masks = [mask_input[:, :, z] for z in range(len(thetas))]
        observer = trainer._observer(dirty, masks, li_list)
        res = forward_backward(m_net, thetas, d_ref, recursion="bidirectional",
                               observer=observer, gain_mode="kalman")
        x_hat_final = trainer._final_output(res, dirty, masks, li_list)

    zi = seq.index(args.sid)
    pred = x_hat_final[0, 0, zi].cpu().numpy()
    tgt = sample["X_clean"][zi, 0].numpy()
    base = sample["x_hat"][zi, 0].numpy()
    msk = sample["metal_mask"][zi, 0].numpy()
    err = np.abs(pred - tgt)

    binm = msk > 0.5
    nonm = ~binm
    rmse = float(np.sqrt(np.mean((pred[nonm] - tgt[nonm]) ** 2)))
    print(f"[check] 非金属区 RMSE={rmse:.1f} HU", flush=True)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6.3))
    for ax, img, t in [(axes[0], tgt, "Target (GT)"),
                       (axes[1], base, "Baseline (artifact)"),
                       (axes[2], pred, "Prediction (v1.4)"),
                       (axes[3], err, "Error |pred-tgt|")]:
        cmap = "gray" if t != "Error |pred-tgt|" else "hot"
        vmin, vmax = (-150, 350) if t != "Error |pred-tgt|" else (None, None)
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(t, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"v1.4 overfit {args.steps} steps | {args.sid} | "
                 f"lambda_ssim={args.lambda_ssim} lambda_grad={args.lambda_grad}\n"
                 f"non-metal RMSE={rmse:.1f} HU", fontsize=12)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {args.out}", flush=True)


if __name__ == "__main__":
    main()
