#!/usr/bin/env python
"""v21_infer_2d.py —— 2D FIND-Net 推理对比：v1.1 vs v2.1（完全收敛的 Stage1）权重。

纯 2D 单切片推理（无 SSM 递归，LIct=经典 LI-MAR，正是 Stage1 训练的对象）：
  Target | Baseline | v1.1 预测 | v2.1 预测
自动挑金属面积最大的前 N 个 train 样本，打印各自非金属区 RMSE。
"""
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

from prism.models.find_net import FINDNet, FINDNetArgs  # noqa: E402

D = "/root/autodl-tmp/PRISM/training_data_astra/train"
META = "/root/autodl-tmp/PRISM/training_data_astra/meta"


def rd(sid, sub):
    return np.fromfile(os.path.join(D, sub, f"{sid}.raw"), np.float32).reshape(512, 512)


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="展示样本数")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/root/autodl-tmp/PRISM/training_data_astra/viz_v12/v21_infer_2d.png")
    args = ap.parse_args()

    v11 = "/root/autodl-tmp/PRISM/training_data_astra/findnet_stage1_v1.1.pt"
    v21 = "/root/autodl-tmp/PRISM/training_data_astra/findnet_stage1_v2.1.pt"

    # 挑金属面积最大的前 N 个 train 样本
    metas = [json.load(open(f)) for f in glob.glob(os.path.join(META, "*.json"))]
    cand = []
    for m in metas:
        if m.get("split") != "train":
            continue
        msk = rd(m["sample_id"], "Mask")
        cand.append(((msk > 0.5).sum(), m["sample_id"]))
    cand.sort(reverse=True)
    sids = [s for _, s in cand[: args.n]]
    print(f"[v21] 展示样本: {sids}", flush=True)

    fn_v11 = FINDNet(FINDNetArgs()).to(args.device)
    fn_v11.load_state_dict(torch.load(v11, map_location=args.device, weights_only=False)["find_net"])
    fn_v21 = FINDNet(FINDNetArgs()).to(args.device)
    fn_v21.load_state_dict(torch.load(v21, map_location=args.device, weights_only=False)["find_net"])
    fn_v11.eval()
    fn_v21.eval()
    print(f"[load] v1.1={v11}  v2.1={v21}", flush=True)

    fig, axes = plt.subplots(args.n, 4, figsize=(22, 5.6 * args.n))
    if args.n == 1:
        axes = axes[None, :]

    def infer(fn, base, li, mask, backfill_r=8):
        b = torch.from_numpy(base[None, None]).to(args.device)
        l = torch.from_numpy(li[None, None]).to(args.device)
        m = torch.from_numpy(mask[None, None]).to(args.device)
        with torch.no_grad():
            x = fn.reconstruct(b, l, m)   # [1,1,H,W] HU（FIND-Net 裸输出）
            # v1.3 金属回填（对齐 SSMTrainer._backfill）：金属区膨胀 r=8 原样保留 Baseline，
            # 杜绝网络在金属植入区填假组织。
            binm = (m > 0.5).float()
            k = 2 * backfill_r + 1
            dil = F.max_pool2d(binm, kernel_size=k, stride=1, padding=backfill_r)
            x = (1.0 - dil) * x + dil * b
            valid = (1.0 - dil).cpu().numpy()[0, 0]   # 1=有效（回填区外）
        return x[0, 0].cpu().numpy(), valid

    for i, sid in enumerate(sids):
        base = rd(sid, "Baseline")
        tgt = rd(sid, "Target")
        li = rd(sid, "LI")
        msk = rd(sid, "Mask")
        p1, valid = infer(fn_v11, base, li, msk)
        p2, _ = infer(fn_v21, base, li, msk)

        # v1.3 评估口径：非金属 且 排除回填晕圈（valid_eval_mask）
        eval_mask = (msk < 0.5) & (valid > 0.5)
        r1 = float(np.sqrt(np.mean((p1[eval_mask] - tgt[eval_mask]) ** 2)))
        r2 = float(np.sqrt(np.mean((p2[eval_mask] - tgt[eval_mask]) ** 2)))
        print(f"[{sid}] 非金属区 RMSE: v1.1={r1:.1f}  v2.1={r2:.1f}  "
              f"(改善 {100*(1-r2/r1):+.1f}%)", flush=True)

        for ax, img, t in [(axes[i, 0], tgt, "Target"),
                           (axes[i, 1], base, "Baseline"),
                           (axes[i, 2], p1, f"v1.1 pred  RMSE={r1:.0f}"),
                           (axes[i, 3], p2, f"v2.1 pred  RMSE={r2:.0f}")]:
            ax.imshow(img, cmap="gray", vmin=-150, vmax=350, interpolation="nearest")
            ax.set_title(t, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[i, 0].set_ylabel(sid, fontsize=10)

    fig.suptitle("2D FIND-Net inference: v1.1 (early-stop epoch 12) vs v2.1 (converged epoch 58)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {args.out}", flush=True)


if __name__ == "__main__":
    main()
