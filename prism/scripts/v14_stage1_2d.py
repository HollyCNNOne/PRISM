#!/usr/bin/env python
"""v14_stage1_2d.py —— findnet_stage1_v1.4.pt 纯 2D 推理验证（不经过 SSM）。

对 sample_000973 用 Stage 1 权重跑单次 FIND-Net 前向（纯 2D），出图：
  Target | Baseline | Prediction（纯 2D v1.4）
  + 鼻窦（上方黑空腔）与颅底骨骼两个 Zoom-in
目的：单独看 SSIM+Grad 雕刻刀在纯 2D 下是否把骨边缘雕锐利、砍掉「浮现感」。
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/autodl-tmp/prism_framework")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prism.models.find_net import FINDNet, FINDNetArgs  # noqa: E402


def rd(path, h=512, w=512):
    return np.fromfile(path, dtype=np.float32).reshape(h, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="/root/autodl-tmp/PRISM/training_data_astra/findnet_stage1_v1.4.pt")
    ap.add_argument("--sid", default="sample_000973")
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out",
                    default="/root/autodl-tmp/PRISM/training_data_astra/viz_v12/v14_stage1_2d.png")
    args = ap.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    base = rd(os.path.join(train_dir, "Baseline", f"{args.sid}.raw"))
    li = rd(os.path.join(train_dir, "LI", f"{args.sid}.raw"))
    msk = rd(os.path.join(train_dir, "Mask", f"{args.sid}.raw"))
    tgt = rd(os.path.join(train_dir, "Target", f"{args.sid}.raw"))

    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    fn = FINDNet(FINDNetArgs()).to(args.device)
    fn.load_state_dict(ck["find_net"], assign=True)   # CPU 需 assign=True（expand 视图参数）
    fn.eval()
    print(f"[load] {args.checkpoint} (epoch {ck.get('config', {}).get('epoch')})", flush=True)

    ct_ma = torch.from_numpy(base)[None, None].float().to(args.device)
    lict = torch.from_numpy(li)[None, None].float().to(args.device)
    mask = torch.from_numpy(msk)[None, None].float().to(args.device)
    with torch.no_grad():
        pred = fn.reconstruct(ct_ma, lict, mask)[0, 0].cpu().numpy()

    # ---- v1.4 出图物理修正（展示层，不改主模型）----
    # 1) 金属回填：Final = (1 − Mask_dilate_r8)·Pred + Mask_dilate_r8·Baseline
    #    —— 丢弃网络在金属区生成的假组织（57.4 HU），恢复 Baseline 金属高光（3000 HU）
    binm = (msk > 0.5).astype(np.float32)
    dil = F.max_pool2d(torch.from_numpy(binm)[None, None],
                       kernel_size=17, stride=1, padding=8)[0, 0].numpy()
    pred = (1.0 - dil) * pred + dil * base
    # 2) 空气钳制：空气区（target < -500）借用 Baseline 纯黑底，杜绝 +41 HU 灰色幽灵
    air = tgt < -500
    pred = np.where(air, base, pred)

    arrs = [tgt, base, pred]
    titles = ["Target (GT)", "Baseline (artifact)", "Prediction (pure 2D v1.4 + physics)"]

    # 全图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, a, t in zip(axes, arrs, titles):
        ax.imshow(a, cmap="gray", vmin=-150, vmax=350, interpolation="nearest")
        ax.set_title(t, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"v1.4 Stage 1 pure 2D | {args.sid}", fontsize=12)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {args.out}", flush=True)

    # 两个 Zoom：鼻窦（上方）+ 颅底（中）
    zooms = [("sinus", 50, 220, 160, 350), ("skull_base", 220, 400, 160, 350)]
    for name, y0, y1, x0, x1 in zooms:
        figz, axz = plt.subplots(1, 3, figsize=(18, 6))
        for ax, a, t in zip(axz, arrs, titles):
            ax.imshow(a[y0:y1, x0:x1], cmap="gray", vmin=-150, vmax=350, interpolation="nearest")
            ax.set_title(t, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
        figz.suptitle(f"v1.4 Stage 1 pure 2D zoom {name} | {args.sid}", fontsize=12)
        zout = args.out.replace(".png", f"_{name}.png")
        figz.savefig(zout, dpi=110, bbox_inches="tight")
        plt.close(figz)
        print(f"[save] {zout}", flush=True)


if __name__ == "__main__":
    main()
