"""viz_four_sizes_correct.py — 四尺寸对比图（正确版：FIND-Net 输出含金属回填）。

布局与 viz_four_sizes.png 相同：
    行 = small / medium / large / giant
    列 = GT | Baseline | FIND-Net(含回填) | |FIND-Net − GT|
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from prism.eval.domain import tissue_mask
import argparse, os, sys
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
R = 8


def rd(sid, sub):
    p = os.path.join(D, "train", sub, f"{sid}.raw")
    if not os.path.exists(p):
        p = os.path.join(D, "val", sub, f"{sid}.raw")
    return np.fromfile(p, np.float32).reshape(512, 512)


def backfill(pred, baseline, mask, r=R):
    b = torch.from_numpy((mask > 0.5).astype(np.float32))[None, None]
    dil = F.max_pool2d(b, kernel_size=2 * r + 1, stride=1, padding=r)[0, 0].numpy()
    return (1.0 - dil) * pred + dil * baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT, "viz_four_sizes_correct.png"))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = FINDNet(FINDNetArgs()).to(dev)
    net.load_state_dict(torch.load(f"{D}/findnet_stage1_v2.1.pt", map_location=dev,
                                   weights_only=False)["find_net"], assign=True)
    net.eval()
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "font.size": 9})
    fig, axes = plt.subplots(len(PICKS), 4, figsize=(14, 13.5))
    for i, (size, sid) in enumerate(PICKS.items()):
        gt, bl, li, mk = (rd(sid, s) for s in ("Target", "Baseline", "LI", "Mask"))
        t = lambda a: torch.from_numpy(a[None, None].astype(np.float32)).to(dev)
        with torch.no_grad():
            raw = net.reconstruct(t(bl), t(li), t(mk))[0, 0].cpu().numpy()
        fn = backfill(raw, bl, mk)
        tis = tissue_mask(mk, gt, r=8)
        print(f"[{size}] Baseline {np.abs(bl-gt)[tis].mean():.1f} -> "
              f"FIND-Net(+backfill) {np.abs(fn-gt)[tis].mean():.1f} HU", flush=True)
        rows = [(gt, "GT", "gray", -200, 400),
                (bl, "Baseline", "gray", -200, 400),
                (fn, "FIND-Net (+ 金属回填)", "gray", -200, 400),
                (np.abs(fn - gt), "|FIND-Net − GT|", "inferno", 0, 300)]
        for j, (img, tt, cm, lo, hi) in enumerate(rows):
            ax = axes[i, j]
            ax.imshow(img, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(tt, fontsize=11)
            if j == 0:
                ax.set_ylabel(size, fontsize=12)
    fig.suptitle("V2.2 four coil sizes — GT / Baseline / FIND-Net(+backfill)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("[save]", args.out)


if __name__ == "__main__":
    main()
