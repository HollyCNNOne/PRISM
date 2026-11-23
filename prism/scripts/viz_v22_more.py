"""viz_v22_more.py — 额外可视化：四尺寸对比 / z 轴连续性条带 / 单尺寸放大。"""
import argparse, glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/prism_framework")
from prism.models.find_net import FINDNet, FINDNetArgs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/root/autodl-tmp/PRISM/training_data_astra"
OUT = "/root/autodl-tmp/PRISM/v2.2"
PICKS = {"small": "sample_003090", "medium": "sample_003169",
         "large": "sample_003260", "giant": "sample_001308"}
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                     "font.size": 9})


def rd(sid, sub):
    p = os.path.join(D, "train", sub, f"{sid}.raw")
    if not os.path.exists(p):
        p = os.path.join(D, "val", sub, f"{sid}.raw")
    return np.fromfile(p, np.float32).reshape(512, 512)


def four_sizes(net, dev):
    fig, axes = plt.subplots(4, 4, figsize=(14, 13.5))
    for i, (size, sid) in enumerate(PICKS.items()):
        gt, bl, li, mk = (rd(sid, s) for s in ("Target", "Baseline", "LI", "Mask"))
        t = lambda a: torch.from_numpy(a[None, None].astype(np.float32)).to(dev)
        with torch.no_grad():
            fn = net.reconstruct(t(bl), t(li), t(mk))[0, 0].cpu().numpy()
        for j, (img, tt, cm, lo, hi) in enumerate([
                (gt, "GT", "gray", -200, 400),
                (bl, "Baseline", "gray", -200, 400),
                (fn, "FIND-Net (v2.1)", "gray", -200, 400),
                (np.abs(fn - gt), "|FIND-Net − GT|", "inferno", 0, 300)]):
            ax = axes[i, j]
            ax.imshow(img, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(tt, fontsize=11)
            if j == 0:
                ax.set_ylabel(size, fontsize=12)
    fig.suptitle("V2.2 four coil sizes — GT / Baseline / FIND-Net", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, "viz_four_sizes.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("[save] viz_four_sizes.png")


def zstrip():
    meta = {}
    for f in glob.glob(os.path.join(D, "meta", "*.json")):
        m = json.load(open(f)); meta[m["sample_id"]] = m
    best = None
    for sid in PICKS.values():
        seq = meta[sid]["sequence_id"]
        sids = sorted([s for s, m in meta.items() if m["sequence_id"] == seq],
                      key=lambda x: int(x.rsplit("_", 1)[1]))
        if len(sids) >= 8 and (best is None or len(sids) > len(best)):
            best = sids
    if not best:
        print("[skip] 无足够长序列"); return
    k = min(8, len(best))
    idx = np.linspace(0, len(best) - 1, k).astype(int)
    sids = [best[i] for i in idx]
    fig, axes = plt.subplots(4, k, figsize=(2.0 * k, 8.4))
    for j, sid in enumerate(sids):
        gt = rd(sid, "Target"); bl = rd(sid, "Baseline")
        mk = rd(sid, "Mask"); dj = np.load(os.path.join(D, "train", "Phys", f"{sid}_d_j.npy"))[0]
        for i, (img, cm, lo, hi) in enumerate([(gt, "gray", -200, 400),
                                               (bl, "gray", -200, 400),
                                               (np.abs(bl - gt), "inferno", 0, 400),
                                               (np.log1p(np.clip(dj, 0, None)), "viridis", None, None)]):
            ax = axes[i, j]
            ax.imshow(img, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(sid.rsplit("_", 1)[1], fontsize=9)
        for i, lab in enumerate(["GT", "Baseline", "|B−GT|", "log d_j"]):
            axes[i, 0].set_ylabel(lab, fontsize=10)
    fig.suptitle(f"z-axis continuity (8 slices of one implant, sequence {meta[sids[0]]['sequence_id']})",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "viz_zstrip.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("[save] viz_zstrip.png")


def main():
    dev = "cuda"
    net = FINDNet(FINDNetArgs()).to(dev)
    net.load_state_dict(torch.load(f"{D}/findnet_stage1_v2.1.pt", map_location=dev,
                                   weights_only=False)["find_net"])
    net.eval()
    four_sizes(net, dev)
    zstrip()


if __name__ == "__main__":
    main()
