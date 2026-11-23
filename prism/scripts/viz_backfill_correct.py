"""viz_backfill_correct.py — 正确可视化：FIND-Net 原始输出 vs 管线最终输出（含金属回填）。

管线契约（trainer._final_output）：
    Final = (1 − dilate(mask>0.5, r=8)) · FIND-Net_out + dilate(...) · Baseline
即金属区（含 8px 晕圈）**永远用 Baseline 覆盖，绝不生成组织**。
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/autodl-tmp/prism_framework")
from prism.models.find_net import FINDNet, FINDNetArgs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/root/autodl-tmp/PRISM/training_data_astra"
OUT = "/root/autodl-tmp/PRISM/v2.2"
PICKS = {"small": "sample_003090", "medium": "sample_003169",
         "large": "sample_003260", "giant": "sample_001308"}
BACKFILL_R = 8


def rd(sid, sub):
    p = os.path.join(D, "train", sub, f"{sid}.raw")
    if not os.path.exists(p):
        p = os.path.join(D, "val", sub, f"{sid}.raw")
    return np.fromfile(p, np.float32).reshape(512, 512)


def backfill(pred, baseline, mask, r=BACKFILL_R):
    b = torch.from_numpy((mask > 0.5).astype(np.float32))[None, None]
    k = 2 * r + 1
    dil = F.max_pool2d(b, kernel_size=k, stride=1, padding=r)[0, 0].numpy()
    return (1.0 - dil) * pred + dil * baseline, dil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT, "viz_final_correct.png"))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = FINDNet(FINDNetArgs()).to(dev)
    net.load_state_dict(torch.load(f"{D}/findnet_stage1_v2.1.pt", map_location=dev,
                                   weights_only=False)["find_net"])
    net.eval()

    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "font.size": 9})
    sizes = list(PICKS)
    fig, axes = plt.subplots(len(sizes), 5, figsize=(17, 3.5 * len(sizes)))
    for i, size in enumerate(sizes):
        sid = PICKS[size]
        gt, bl, li, mk = (rd(sid, s) for s in ("Target", "Baseline", "LI", "Mask"))
        t = lambda a: torch.from_numpy(a[None, None].astype(np.float32)).to(dev)
        with torch.no_grad():
            raw = net.reconstruct(t(bl), t(li), t(mk))[0, 0].cpu().numpy()
        final, dil = backfill(raw, bl, mk)
        tis = (mk <= 1e-3)
        print(f"[{size}] metal_sum={mk.sum():.1f}  "
              f"raw err {np.abs(raw-gt)[tis].mean():.1f}  "
              f"final err {np.abs(final-gt)[tis].mean():.1f} HU", flush=True)
        rows = [
            (gt, "GT", "gray", -200, 400),
            (bl, "Baseline", "gray", -200, 400),
            (raw, "FIND-Net RAW (无回填)", "gray", -200, 400),
            (final, "FIND-Net + 金属回填（管线真实输出）", "gray", -200, 400),
            (dil, "回填掩膜 dilate(mask>0.5, 8px)", "gray", 0, 1),
        ]
        for j, (img, tt, cm, lo, hi) in enumerate(rows):
            ax = axes[i, j]
            ax.imshow(img, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(tt, fontsize=9)
            if j == 0:
                ax.set_ylabel(size, fontsize=12)
    fig.suptitle("V2.2 — FIND-Net raw vs pipeline final (metal backfill r=8)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("[save]", args.out)


if __name__ == "__main__":
    main()
