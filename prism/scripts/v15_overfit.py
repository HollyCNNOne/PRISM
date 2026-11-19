#!/usr/bin/env python
"""v15_overfit.py —— v1.5 方向性梯度解耦 50 步 overfit + v1.3/v1.5 对比图。

训练 50 步（v1.5 loss：loss_gx+loss_gy 显式拆出），出图：
  Target | v1.3 Prediction | v1.5 Prediction | Error(v1.5)
+ 过渡区 / 阴影区 / 充分区三个 Zoom。同时打印 MSE/SSIM/Grad 下降曲线。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

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


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck["config"]
    m_net = MNetwork(nb=cfg.get("nb", 1), channels=cfg.get("channels", 32),
                     use_velocity=cfg.get("use_velocity", True)).to(device)
    find_net = FINDNet(FINDNetArgs()).to(device)
    m_net.load_state_dict(ck["model"], assign=True)
    find_net.load_state_dict(ck["find_net"], assign=True)
    m_net.eval()
    find_net.eval()
    config = SSMConfig(nb=cfg.get("nb", 1), channels=cfg.get("channels", 32),
                       recursion=cfg.get("recursion", "bidirectional"), d_ref=cfg["d_ref"],
                       gain_mode=cfg.get("gain_mode", "kalman"),
                       findnet_lict_source=cfg.get("findnet_lict_source", "mu"),
                       binarize_mask=cfg.get("binarize_mask", False),
                       mask_bin_thr=cfg.get("mask_bin_thr", 0.5),
                       backfill_r=cfg.get("backfill_r", 8))
    return SSMTrainer(m_net, config, device=device, find_net=find_net), config


def run_infer(trainer, config, thetas, x_hat, li, metal_mask):
    Z = len(thetas)
    dirty_vol = x_hat.permute(0, 2, 1, 3, 4)
    dirty = [dirty_vol[:, :, z] for z in range(Z)]
    li_vol = li.permute(0, 2, 1, 3, 4)
    li_list = [li_vol[:, :, z] for z in range(Z)]
    masks = [metal_mask[:, :, z] for z in range(Z)]
    observer = trainer._observer(dirty, masks, li_list)
    with torch.no_grad():
        res = forward_backward(trainer.m_net, thetas, config.d_ref,
                               recursion=config.recursion, observer=observer,
                               gain_mode=config.gain_mode)
        return trainer._final_output(res, dirty, masks, li_list)  # [1,1,Z,H,W]


def dilate(binm, r):
    t = torch.from_numpy(binm.astype(np.float32))[None, None]
    k = 2 * r + 1
    return F.max_pool2d(t, kernel_size=k, stride=1, padding=r)[0, 0].numpy() > 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt13", default="/root/autodl-tmp/PRISM/training_data_astra/ablations_v13/00_baseline.pt")
    ap.add_argument("--sid", default="sample_000973")
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--resume-findnet", default="/root/autodl-tmp/PRISM/training_data_astra/findnet_stage1_v1.4.pt")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda-ssim", type=float, default=0.25)
    ap.add_argument("--lambda-grad", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="/root/autodl-tmp/PRISM/training_data_astra/viz_v12")
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
    print(f"[v15] sid={args.sid} seq={len(seq)} lambda_ssim={args.lambda_ssim} lambda_grad={args.lambda_grad}",
          flush=True)

    loader = make_theta_sequence_loader(
        os.path.join(train_dir, "Target"), os.path.join(train_dir, "Baseline"),
        os.path.join(train_dir, "LI"), os.path.join(train_dir, "Mask"),
        os.path.join(train_dir, "Phys"))
    sample = loader(seq)
    thetas = [{k: v.unsqueeze(0).to(args.device) for k, v in t.items()} for t in sample["thetas"]]
    X_clean = sample["X_clean"].unsqueeze(0).to(args.device)
    metal_mask = sample["metal_mask"].unsqueeze(0).to(args.device)
    x_hat = sample["x_hat"].unsqueeze(0).to(args.device)
    li = sample["li"].unsqueeze(0).to(args.device)

    # v1.5 模型（fresh MNetwork + warm-start FINDNet）
    m_net = MNetwork(nb=1, channels=32, use_velocity=True).to(args.device)
    find_net = FINDNet(FINDNetArgs()).to(args.device)
    if os.path.exists(args.resume_findnet):
        ck = torch.load(args.resume_findnet, map_location=args.device, weights_only=False)
        find_net.load_state_dict(ck["find_net"])
    config = SSMConfig(nb=1, channels=32, recursion="bidirectional", d_ref=4.10023e+06,
                       cold_start_init="obs", lambda_q=1.0, lambda_pc=0.1,
                       lambda_down=1.0, lambda_mu=1.0,
                       lambda_ssim=args.lambda_ssim, lambda_grad=args.lambda_grad, backfill_r=8)
    trainer = SSMTrainer(m_net, config, device=args.device, find_net=find_net)
    opt = torch.optim.Adam([{"params": m_net.parameters(), "lr": args.lr},
                            {"params": find_net.parameters(), "lr": args.lr}], lr=args.lr)
    trainer.optimizer = opt

    print(f"{'step':>4} {'total':>9} {'MSE':>8} {'SSIM':>8} {'Grad':>8}", flush=True)
    for step in range(args.steps):
        total, q, pc, down, mu = trainer.train_step(
            thetas, X_clean, metal_mask, x_hat, clip_grad=0.0, li=li)
        mse, ssim, grad = trainer.last_down_parts
        print(f"{step+1:4d} {total.item():9.3f} {mse.item():8.3f} {ssim.item():8.3f} {grad.item():8.3f}",
              flush=True)

    # v1.5 推理
    m_net.eval()
    find_net.eval()
    mm = metal_mask.permute(0, 2, 1, 3, 4)
    x15 = run_infer(trainer, config, thetas, x_hat, li, mm)

    # v1.3 推理
    trainer13, cfg13 = load_model(args.ckpt13, args.device)
    x13 = run_infer(trainer13, cfg13, thetas, x_hat, li, mm)

    zi = seq.index(args.sid)
    pred15 = x15[0, 0, zi].cpu().numpy()
    pred13 = x13[0, 0, zi].cpu().numpy()
    tgt = sample["X_clean"][zi, 0].numpy()
    msk = sample["metal_mask"][zi, 0].numpy()
    d_j = np.load(os.path.join(train_dir, "Phys", f"{args.sid}_d_j.npy")).reshape(512, 512)
    d_ref = 4.10023e+06
    core = msk > 0.5
    r = d_j / d_ref
    backfill = dilate(core, 8)
    trans = (dilate(core, 18) & ~dilate(core, 8)) & (r < 1 / 9)
    shadow = (~core) & (~backfill) & (r < 1 / 9)
    suff = (~core) & (~backfill) & (r > 9)
    for name, m in [("过渡区", trans), ("完全阴影", shadow), ("充分区", suff)]:
        rm13 = float(np.sqrt(np.mean((pred13[m] - tgt[m]) ** 2))) if m.sum() > 0 else float("nan")
        rm15 = float(np.sqrt(np.mean((pred15[m] - tgt[m]) ** 2))) if m.sum() > 0 else float("nan")
        print(f"[{name}] v1.3={rm13:.1f}  v1.5={rm15:.1f}  (px={int(m.sum())})", flush=True)

    err15 = np.abs(pred15 - tgt)
    arrs = [tgt, pred13, pred15, err15]
    titles = ["Target (GT)", "v1.3 Prediction", "v1.5 Prediction", "Error(v1.5)"]
    ys, xs = np.nonzero(core)
    cy, cx = int(ys.mean()), int(xs.mean())

    def save(fig, out):
        os.makedirs(args.outdir, exist_ok=True)
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[save] {out}", flush=True)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6.3))
    for ax, a, t in zip(axes, arrs, titles):
        cmap, vmin, vmax = ("hot", None, None) if "Error" in t else ("gray", -150, 350)
        ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(t, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"v1.3 vs v1.5 (50-step overfit) | {args.sid}", fontsize=13)
    save(fig, os.path.join(args.outdir, "v15_vs_v13_full.png"))

    for name, y0, y1, x0, x1 in [("transition", cy - 56, cy + 56, cx - 56, cx + 56),
                                 ("shadow", cy - 100, cy + 100, cx - 100, cx + 100),
                                 ("sufficient", 60, 200, 160, 350)]:
        y0, y1, x0, x1 = max(0, y0), min(512, y1), max(0, x0), min(512, x1)
        fig, axes = plt.subplots(1, 4, figsize=(24, 6.3))
        for ax, a, t in zip(axes, arrs, titles):
            cmap, vmin, vmax = ("hot", None, None) if "Error" in t else ("gray", -150, 350)
            ax.imshow(a[y0:y1, x0:x1], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(t, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"v1.3 vs v1.5 zoom {name} | {args.sid}", fontsize=13)
        save(fig, os.path.join(args.outdir, f"v15_vs_v13_{name}.png"))


if __name__ == "__main__":
    main()
