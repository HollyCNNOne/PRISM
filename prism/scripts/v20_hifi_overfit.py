#!/usr/bin/env python
"""v20_hifi_overfit.py —— HiFi 刻画块接入 FIND-Net 的快速 overfit 对照（v2.0 方向）。

对照 FIND-Net 尾端接不接 HiFi 刻画块（use_hifi 0/1），同一序列、同 seed、同 warm-start、
同步数，看：
  1. 非金属区 RMSE（整体）；
  2. 完全阴影区 RMSE（d_j/d_ref < 1/9，金属核后=盲区=高频模糊重灾区）；
  3. 视觉：Target | Baseline | Prediction（4 列，末列误差热图）。

用法（先跑 hifi=0 再跑 hifi=1）：
  python v20_hifi_overfit.py --hifi 0 --steps 120 --out .../v20_hifi0.png
  python v20_hifi_overfit.py --hifi 1 --steps 120 --out .../v20_hifi1.png
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
import torch.nn.functional as F

from prism.data.dataset import make_theta_sequence_loader  # noqa: E402
from prism.models.m_network import MNetwork  # noqa: E402
from prism.models.find_net import FINDNet, FINDNetArgs  # noqa: E402
from prism.recursion.algorithm1 import forward_backward  # noqa: E402
from prism.training.trainer import SSMTrainer  # noqa: E402
from prism.utils.config import SSMConfig  # noqa: E402

D_REF = 4.10023e+06


def _sample_idx(sid):
    return int(sid.rsplit("_", 1)[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", default="sample_000973")
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--resume-findnet",
                    default="/root/autodl-tmp/PRISM/training_data_astra/findnet_stage1_v1.1.pt")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--hifi", type=int, default=0, help="1=在 FIND-Net 尾端接 HiFi 刻画块")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/root/autodl-tmp/PRISM/training_data_astra/viz_v12/v20_hifi.png")
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
    print(f"[v20] sid={args.sid} seq={len(seq)} slices hifi={args.hifi}", flush=True)

    loader = make_theta_sequence_loader(
        os.path.join(train_dir, "Target"), os.path.join(train_dir, "Baseline"),
        os.path.join(train_dir, "LI"), os.path.join(train_dir, "Mask"),
        os.path.join(train_dir, "Phys"))

    m_net = MNetwork(nb=1, channels=32, use_velocity=True).to(args.device)
    find_net = FINDNet(FINDNetArgs(use_hifi=bool(args.hifi))).to(args.device)
    if args.resume_findnet and os.path.exists(args.resume_findnet):
        ck = torch.load(args.resume_findnet, map_location=args.device, weights_only=False)
        # strict=False：warm-start 无 hifi 块权重，hifi 块随机 init（初始恒等），其余加载
        missing, unexpected = find_net.load_state_dict(ck["find_net"], strict=False)
        print(f"[load] find_net from {args.resume_findnet} "
              f"(missing={len(missing)} unexpected={len(unexpected)})", flush=True)

    config = SSMConfig(nb=1, channels=32, recursion="bidirectional", d_ref=D_REF,
                       cold_start_init="obs", lambda_q=1.0, lambda_pc=0.1,
                       lambda_down=1.0, lambda_mu=1.0,
                       lambda_ssim=0.1, backfill_r=8)
    trainer = SSMTrainer(m_net, config, device=args.device, find_net=find_net)
    opt = torch.optim.Adam([
        {"params": m_net.parameters(), "lr": args.lr},
        {"params": find_net.parameters(), "lr": args.lr},
    ], lr=args.lr)
    trainer.optimizer = opt

    sample = loader(seq)

    # GPU 被 v2.1 Stage1 占用（剩 ~2.5GB），512×512 overfit 训练步需 ~4.75GB 会 OOM；
    # 降到 256×256（内存 ÷4）。对照实验（hifi 0 vs 1，同分辨率同 seed）依然成立。
    def _ds(t, factor=2):
        lead = t.shape[:-2]
        h, w = t.shape[-2], t.shape[-1]
        return F.avg_pool2d(t.reshape(-1, 1, h, w), kernel_size=factor, stride=factor).reshape(
            *lead, h // factor, w // factor)

    sample["X_clean"] = _ds(sample["X_clean"])
    sample["x_hat"] = _ds(sample["x_hat"])
    sample["metal_mask"] = _ds(sample["metal_mask"])
    sample["li"] = _ds(sample["li"])
    for t in sample["thetas"]:
        for k in list(t.keys()):
            t[k] = _ds(t[k])
    print(f"[v20] 下采样到 256×256（GPU 只剩 ~2.5GB，避开 v2.1 训练）", flush=True)

    thetas = [{k: v.unsqueeze(0).to(args.device) for k, v in t.items()} for t in sample["thetas"]]
    X_clean = sample["X_clean"].unsqueeze(0).to(args.device)      # [1,Z,1,H,W]
    metal_mask = sample["metal_mask"].unsqueeze(0).to(args.device)
    x_hat = sample["x_hat"].unsqueeze(0).to(args.device)
    li = sample["li"].unsqueeze(0).to(args.device)

    # 完全阴影区掩膜（盲区）：d_j/d_ref < 1/9 且非金属
    d_j = torch.cat([t["d_j"] for t in thetas], dim=0)          # [Z,1,H,W]（每片 [1,1,H,W]）
    r = d_j / D_REF
    blind = ((r < 1.0 / 9.0).float() * (1.0 - metal_mask[0]).float())  # [Z,1,H,W]

    for step in range(args.steps):
        total, q, pc, down, mu = trainer.train_step(
            thetas, X_clean, metal_mask, x_hat, clip_grad=0.0, li=li)
        if (step + 1) % 30 == 0 or step + 1 == args.steps:
            print(f"[step {step+1}/{args.steps}] total={total.item():.4f} down={down.item():.4f} "
                  f"mu={mu.item():.4f}", flush=True)

    # ---- 推理 ----
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
        res = forward_backward(m_net, thetas, D_REF, recursion="bidirectional",
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
    # 完全阴影区（盲区）RMSE —— 高频模糊重灾区
    bm = blind[zi, 0].cpu().numpy() > 0.5
    blind_rmse = float(np.sqrt(np.mean((pred[bm] - tgt[bm]) ** 2))) if bm.sum() > 0 else float("nan")
    print(f"[check] hifi={args.hifi} 非金属区 RMSE={rmse:.1f} HU | "
          f"完全阴影区 RMSE={blind_rmse:.1f} HU (盲区像素={bm.sum()})", flush=True)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6.3))
    for ax, img, t in [(axes[0], tgt, "Target (GT)"),
                       (axes[1], base, "Baseline (artifact)"),
                       (axes[2], pred, f"Prediction (hifi={args.hifi})")]:
        ax.imshow(img, cmap="gray", vmin=-150, vmax=350, interpolation="nearest")
        ax.set_title(t, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
    im3 = axes[3].imshow(err, cmap="hot", interpolation="nearest")
    axes[3].set_title(f"Error |pred-tgt| max={err.max():.0f} HU", fontsize=14)
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    fig.colorbar(im3, ax=axes[3], fraction=0.046)
    fig.suptitle(f"v2.0-HiFi overfit {args.steps} steps | {args.sid} | hifi={args.hifi}\n"
                 f"non-metal RMSE={rmse:.1f} | blind RMSE={blind_rmse:.1f} HU", fontsize=12)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {args.out}", flush=True)


if __name__ == "__main__":
    main()
