#!/usr/bin/env python
"""v14_vs_v13_viz.py —— v1.3 vs v1.4 视觉验伤对比（充分区/完全阴影/过渡区）。

对 sample_000973 分别用 v1.3 与 v1.4 权重跑前向，出 4 列图：
  Target | v1.3 Prediction | v1.4 Prediction | Error(v1.4)
并出三个 Zoom：充分区（骨边缘）、完全阴影区（死黑区是否造假）、过渡区（金属边缘亮环）。
同时打印三个区域的单切片 RMSE（v1.3 vs v1.4）佐证。
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
    nb = cfg.get("nb", 1)
    channels = cfg.get("channels", 32)
    use_velocity = cfg.get("use_velocity", True)
    recursion = cfg.get("recursion", "bidirectional")
    gain_mode = cfg.get("gain_mode", "kalman")
    findnet_lict = cfg.get("findnet_lict_source", "mu")
    binarize = cfg.get("binarize_mask", False)
    mask_bin_thr = cfg.get("mask_bin_thr", 0.5)
    backfill_r = cfg.get("backfill_r", 8)
    m_net = MNetwork(nb=nb, channels=channels, use_velocity=use_velocity).to(device)
    find_net = FINDNet(FINDNetArgs()).to(device)
    m_net.load_state_dict(ck["model"], assign=True)
    find_net.load_state_dict(ck["find_net"], assign=True)
    m_net.eval()
    find_net.eval()
    config = SSMConfig(nb=nb, channels=channels, recursion=recursion, d_ref=cfg["d_ref"],
                       gain_mode=gain_mode, findnet_lict_source=findnet_lict,
                       binarize_mask=binarize, mask_bin_thr=mask_bin_thr, backfill_r=backfill_r)
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
        x_hat_final = trainer._final_output(res, dirty, masks, li_list)  # [1,1,Z,H,W]
    return x_hat_final


def dilate(binm, r):
    t = torch.from_numpy(binm.astype(np.float32))[None, None]
    k = 2 * r + 1
    return F.max_pool2d(t, kernel_size=k, stride=1, padding=r)[0, 0].numpy() > 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt13",
                    default="/root/autodl-tmp/PRISM/training_data_astra/ablations_v13/00_baseline.pt")
    ap.add_argument("--ckpt14",
                    default="/root/autodl-tmp/PRISM/training_data_astra/ablations_v14/00_baseline.pt")
    ap.add_argument("--sid", default="sample_000973")
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir",
                    default="/root/autodl-tmp/PRISM/training_data_astra/viz_v12")
    args = ap.parse_args()

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
    print(f"[viz] sid={args.sid} seq={len(seq)} slices", flush=True)

    loader = make_theta_sequence_loader(
        os.path.join(train_dir, "Target"), os.path.join(train_dir, "Baseline"),
        os.path.join(train_dir, "LI"), os.path.join(train_dir, "Mask"),
        os.path.join(train_dir, "Phys"))
    sample = loader(seq)
    thetas = [{k: v.unsqueeze(0).to(args.device) for k, v in t.items()} for t in sample["thetas"]]
    x_hat = sample["x_hat"].unsqueeze(0).to(args.device)
    metal_mask = sample["metal_mask"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(args.device)
    li = sample["li"].unsqueeze(0).to(args.device)

    trainer13, cfg13 = load_model(args.ckpt13, args.device)
    trainer14, cfg14 = load_model(args.ckpt14, args.device)
    x13 = run_infer(trainer13, cfg13, thetas, x_hat, li, metal_mask)
    x14 = run_infer(trainer14, cfg14, thetas, x_hat, li, metal_mask)

    zi = seq.index(args.sid)
    pred13 = x13[0, 0, zi].cpu().numpy()
    pred14 = x14[0, 0, zi].cpu().numpy()
    tgt = sample["X_clean"][zi, 0].numpy()
    base = sample["x_hat"][zi, 0].numpy()
    msk = sample["metal_mask"][zi, 0].numpy()
    d_j = np.load(os.path.join(train_dir, "Phys", f"{args.sid}_d_j.npy")).reshape(512, 512)
    d_ref = cfg14.d_ref

    core = msk > 0.5
    r = d_j / d_ref
    nonmetal = ~core
    backfill = dilate(core, cfg14.backfill_r)
    valid = ~backfill
    # 三个区域掩膜
    trans = (dilate(core, cfg14.backfill_r + 10) & ~dilate(core, cfg14.backfill_r)) & (r < 1 / 9)
    shadow = nonmetal & valid & (r < 1 / 9)
    suff = nonmetal & valid & (r > 9)

    for name, m in [("过渡区", trans), ("完全阴影", shadow), ("充分区", suff)]:
        rmse13 = float(np.sqrt(np.mean((pred13[m] - tgt[m]) ** 2))) if m.sum() > 0 else float("nan")
        rmse14 = float(np.sqrt(np.mean((pred14[m] - tgt[m]) ** 2))) if m.sum() > 0 else float("nan")
        print(f"[{name}] v1.3 RMSE={rmse13:.1f}  v1.4 RMSE={rmse14:.1f}  (像素={int(m.sum())})",
              flush=True)

    err14 = np.abs(pred14 - tgt)
    ys, xs = np.nonzero(core)
    cy, cx = int(ys.mean()), int(xs.mean())

    arrs = [tgt, pred13, pred14, err14]
    titles = ["Target (GT)", "v1.3 Prediction", "v1.4 Prediction", "Error(v1.4)"]

    def save(fig, out):
        os.makedirs(args.outdir, exist_ok=True)
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[save] {out}", flush=True)

    # 全图
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.3))
    for ax, a, t in zip(axes, arrs, titles):
        cmap, vmin, vmax = ("hot", None, None) if "Error" in t else ("gray", -150, 350)
        ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(t, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"v1.3 vs v1.4 | {args.sid}", fontsize=13)
    save(fig, os.path.join(args.outdir, "v14_vs_v13_full.png"))

    # 三个 Zoom
    zooms = [("transition", cy - 56, cy + 56, cx - 56, cx + 56),
             ("shadow", cy - 100, cy + 100, cx - 100, cx + 100),
             ("sufficient", 60, 200, 160, 350)]
    for name, y0, y1, x0, x1 in zooms:
        y0, y1 = max(0, y0), min(512, y1)
        x0, x1 = max(0, x0), min(512, x1)
        fig, axes = plt.subplots(1, 4, figsize=(24, 6.3))
        for ax, a, t in zip(axes, arrs, titles):
            cmap, vmin, vmax = ("hot", None, None) if "Error" in t else ("gray", -150, 350)
            ax.imshow(a[y0:y1, x0:x1], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(t, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"v1.3 vs v1.4 zoom {name} | {args.sid}", fontsize=13)
        save(fig, os.path.join(args.outdir, f"v14_vs_v13_{name}.png"))


if __name__ == "__main__":
    main()
