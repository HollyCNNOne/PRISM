#!/usr/bin/env python
"""v13_overfit.py —— v1.3 修复单序列过拟合验证（sample_000973）。

验证 v1.3 两项修复：
  1. 金属回填：Final = (1−Mask_dilate_r8)·Pred + Mask_dilate_r8·Baseline（金属高光保留）。
  2. SSIM 高频约束：L_down = masked_MSE + λ_ssim·(1−SSIM)（边缘锐利、不糊）。

warm-start findnet_stage1_v1.1.pt，对 sample_000973 所在序列跑 N 步 A-step 全损失，
末步出 4 列图（Target|Baseline|Prediction|Error），前三列强制 vmin=-150/vmax=350。
同时打印金属区 HU min/max（应 >3000，验证回填）与非金属区 RMSE。
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
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda-ssim", type=float, default=0.1)
    ap.add_argument("--backfill-r", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/root/autodl-tmp/PRISM/training_data_astra/viz_v12/v13_overfit.png")
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
    assert seq_id is not None, f"未找到 {args.sid}"
    seq = sorted([json.load(open(f))["sample_id"] for f in glob.glob(os.path.join(meta_dir, "*.json"))
                  if json.load(open(f)).get("sequence_id") == seq_id], key=_sample_idx)
    print(f"[v13] sid={args.sid} seq={len(seq)} slices lambda_ssim={args.lambda_ssim} "
          f"backfill_r={args.backfill_r}", flush=True)

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
                       lambda_ssim=args.lambda_ssim, backfill_r=args.backfill_r)
    trainer = SSMTrainer(m_net, config, device=args.device, find_net=find_net)
    opt = torch.optim.Adam([
        {"params": m_net.parameters(), "lr": args.lr},
        {"params": find_net.parameters(), "lr": args.lr},
    ], lr=args.lr)
    trainer.optimizer = opt

    sample = loader(seq)
    thetas = [{k: v.unsqueeze(0).to(args.device) for k, v in t.items()} for t in sample["thetas"]]
    X_clean = sample["X_clean"].unsqueeze(0).to(args.device)      # [1,Z,1,H,W]
    metal_mask = sample["metal_mask"].unsqueeze(0).to(args.device)  # [1,Z,1,H,W]
    x_hat = sample["x_hat"].unsqueeze(0).to(args.device)          # [1,Z,1,H,W]
    li = sample["li"].unsqueeze(0).to(args.device)

    for step in range(args.steps):
        total, q, pc, down, mu = trainer.train_step(
            thetas, X_clean, metal_mask, x_hat, clip_grad=0.0, li=li)
        if (step + 1) % 20 == 0 or step + 1 == args.steps:
            print(f"[step {step+1}/{args.steps}] total={total.item():.4f} down={down.item():.4f} "
                  f"mu={mu.item():.4f} q={q.item():.4f}", flush=True)

    # ---- 推理出图（复用 eval_mar 的前向路径）----
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
        x_hat_final = trainer._final_output(res, dirty, masks, li_list)  # [1,1,Z,H,W]

    zi = seq.index(args.sid)
    pred = x_hat_final[0, 0, zi].cpu().numpy()
    tgt = sample["X_clean"][zi, 0].numpy()
    base = sample["x_hat"][zi, 0].numpy()
    msk = sample["metal_mask"][zi, 0].numpy()
    err = np.abs(pred - tgt)

    binm = msk > 0.5
    nonm = ~binm
    rmse = float(np.sqrt(np.mean((pred[nonm] - tgt[nonm]) ** 2)))
    if binm.any():
        metal_hu = pred[binm]
        print(f"[check] 金属区 pred min/max={metal_hu.min():.0f}/{metal_hu.max():.0f} HU "
              f"(应 >3000，验证回填保留金属高光)", flush=True)
    print(f"[check] 非金属区 RMSE={rmse:.1f} HU", flush=True)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6.3))
    for ax, img, t in [(axes[0], tgt, "Target (GT)"),
                       (axes[1], base, "Baseline (artifact)"),
                       (axes[2], pred, "Prediction (v1.3 backfill)")]:
        ax.imshow(img, cmap="gray", vmin=-150, vmax=350, interpolation="nearest")
        ax.set_title(t, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
    im3 = axes[3].imshow(err, cmap="hot", interpolation="nearest")
    axes[3].set_title(f"Error |pred-tgt|  max={err.max():.0f} HU", fontsize=14)
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    fig.colorbar(im3, ax=axes[3], fraction=0.046)
    fig.suptitle(
        f"v1.3 overfit {args.steps} steps | {args.sid} | lambda_ssim={args.lambda_ssim} "
        f"backfill_r={args.backfill_r}\n"
        f"non-metal RMSE={rmse:.1f} HU | metal HU max={pred[binm].max() if binm.any() else 0:.0f}",
        fontsize=12)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {args.out}", flush=True)


if __name__ == "__main__":
    main()
