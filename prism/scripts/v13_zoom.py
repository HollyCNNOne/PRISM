#!/usr/bin/env python
"""v13_zoom.py —— v1.3 视觉验尸：backfill_r=8 vs r=0 的金属边缘 Zoom-in 对比。

加载 v1.3 00_baseline 权重，对 sample_000973 所在序列跑一次前向（不训练），
同一 res 上分别用 backfill_r=8 与 backfill_r=0 出最终输出，出图：
  上行：Target | Baseline | Prediction(r=8) | Prediction(r=0)（全图，vmin=-150/vmax=350）
  下行：金属边缘局部 Zoom-in（金属质心 ±64px ROI，同一窗口四张）
并打印金属 10px 环带 RMSE（r=8 vs r=0）做定量佐证。
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


def dilate_np(binm, r):
    """binm: [H,W] bool → [H,W] float32（max_pool 膨胀 r px）。"""
    t = torch.from_numpy(binm.astype(np.float32))[None, None]
    k = 2 * r + 1
    return F.max_pool2d(t, kernel_size=k, stride=1, padding=r)[0, 0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="/root/autodl-tmp/PRISM/training_data_astra/ablations_v13/00_baseline.pt")
    ap.add_argument("--sid", default="sample_000973")
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/root/autodl-tmp/PRISM/training_data_astra/viz_v12/v13_zoom.png")
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

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = ckpt["config"]
    d_ref = cfg["d_ref"]
    nb = cfg.get("nb", 1)
    channels = cfg.get("channels", 32)
    use_velocity = cfg.get("use_velocity", True)
    recursion = cfg.get("recursion", "bidirectional")
    gain_mode = cfg.get("gain_mode", "kalman")
    findnet_lict_source = cfg.get("findnet_lict_source", "mu")
    backfill_r = cfg.get("backfill_r", 8)

    m_net = MNetwork(nb=nb, channels=channels, use_velocity=use_velocity).to(args.device)
    find_net = FINDNet(FINDNetArgs()).to(args.device)
    m_net.load_state_dict(ckpt["model"], assign=True)
    find_net.load_state_dict(ckpt["find_net"], assign=True)
    m_net.eval()
    find_net.eval()

    def make_cfg(r):
        return SSMConfig(nb=nb, channels=channels, recursion=recursion, d_ref=d_ref,
                         gain_mode=gain_mode, findnet_lict_source=findnet_lict_source,
                         backfill_r=r)

    trainer = SSMTrainer(m_net, make_cfg(backfill_r), device=args.device, find_net=find_net)
    loader = make_theta_sequence_loader(
        os.path.join(train_dir, "Target"), os.path.join(train_dir, "Baseline"),
        os.path.join(train_dir, "LI"), os.path.join(train_dir, "Mask"),
        os.path.join(train_dir, "Phys"))

    sample = loader(seq)
    thetas = [{k: v.unsqueeze(0).to(args.device) for k, v in t.items()} for t in sample["thetas"]]
    x_hat = sample["x_hat"].unsqueeze(0).to(args.device)
    metal_mask = sample["metal_mask"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(args.device)
    li = sample["li"].unsqueeze(0).to(args.device)
    Z = len(thetas)
    dirty_vol = x_hat.permute(0, 2, 1, 3, 4)
    dirty = [dirty_vol[:, :, z] for z in range(Z)]
    li_vol = li.permute(0, 2, 1, 3, 4)
    li_list = [li_vol[:, :, z] for z in range(Z)]
    masks = [metal_mask[:, :, z] for z in range(Z)]
    observer = trainer._observer(dirty, masks, li_list)

    with torch.no_grad():
        res = forward_backward(m_net, thetas, d_ref, recursion=recursion,
                               observer=observer, gain_mode=gain_mode)
        x8 = trainer._final_output(res, dirty, masks, li_list)   # backfill_r=8
        trainer.config = make_cfg(0)
        x0 = trainer._final_output(res, dirty, masks, li_list)   # backfill_r=0

    zi = seq.index(args.sid)
    pred8 = x8[0, 0, zi].cpu().numpy()
    pred0 = x0[0, 0, zi].cpu().numpy()
    tgt = sample["X_clean"][zi, 0].numpy()
    base = sample["x_hat"][zi, 0].numpy()
    msk = sample["metal_mask"][zi, 0].numpy()
    binm = msk > 0.5

    # 金属 10px 环带 RMSE（eval 过渡区的简化单切片版）
    ring10 = (dilate_np(binm, 10) - dilate_np(binm, 0)) > 0
    rmse8 = float(np.sqrt(np.mean((pred8[ring10] - tgt[ring10]) ** 2)))
    rmse0 = float(np.sqrt(np.mean((pred0[ring10] - tgt[ring10]) ** 2)))
    print(f"[ring10] backfill_r=8 RMSE={rmse8:.1f} HU  |  backfill_r=0 RMSE={rmse0:.1f} HU", flush=True)
    if binm.any():
        ys, xs = np.nonzero(binm)
        cy, cx = int(ys.mean()), int(xs.mean())
        print(f"[metal] centroid=({cy},{cx}) 金属核 HU(base)={base[binm].mean():.0f}", flush=True)
    else:
        cy, cx = 256, 256

    # Zoom ROI（金属质心 ±64）
    r = 64
    y0, y1 = max(0, cy - r), min(512, cy + r)
    x0, x1 = max(0, cx - r), min(512, cx + r)

    arrs = [tgt, base, pred8, pred0]
    titles = ["Target (GT)", "Baseline (artifact)", "Prediction (r=8)", "Prediction (r=0)"]
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    for c, (a, t) in enumerate(zip(arrs, titles)):
        axes[0, c].imshow(a, cmap="gray", vmin=-150, vmax=350, interpolation="nearest")
        axes[0, c].set_title(t, fontsize=13)
        axes[1, c].imshow(a[y0:y1, x0:x1], cmap="gray", vmin=-150, vmax=350, interpolation="nearest")
        axes[1, c].set_title(f"zoom {t}", fontsize=11)
        for row in (0, 1):
            axes[row, c].set_xticks([])
            axes[row, c].set_yticks([])
    fig.suptitle(
        f"v1.3 zoom | {args.sid} | metal-edge ROI\n"
        f"metal 10px ring RMSE: r=8 -> {rmse8:.0f} HU | r=0 -> {rmse0:.0f} HU",
        fontsize=13)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {args.out}", flush=True)


if __name__ == "__main__":
    main()
