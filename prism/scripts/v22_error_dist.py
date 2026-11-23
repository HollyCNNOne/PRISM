"""v22_error_dist.py — 误差分布分解（发现3 的实证）：四尺寸 × 三状态。

不看均值，看**误差分布的形状**：按 |err| 分箱统计占比，并给分位数。
目的：验证 large/giant 上「MAE 变差、>50HU 变好」是否源于
      「中等错误清掉 + 极端错误放大」。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from prism.eval.domain import tissue_mask
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/autodl-tmp/prism_framework")
from prism.models.find_net import FINDNet, FINDNetArgs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/root/autodl-tmp/PRISM/training_data_astra"
CACHE = "/root/autodl-tmp/PRISM/v2.2/info_loss/_before_cache"
OUT = "/root/autodl-tmp/PRISM/v2.2/info_loss"
PICKS = {"small": "sample_003090", "medium": "sample_003169",
         "large": "sample_003260", "giant": "sample_001308"}
EDGES = [0, 50, 100, 200, 400, 1e9]
LABELS = ["<50", "50-100", "100-200", "200-400", ">400"]


def rd(sid, sub):
    for sp in ("train", "val"):
        p = os.path.join(D, sp, sub, f"{sid}.raw")
        if os.path.exists(p):
            return np.fromfile(p, np.float32).reshape(512, 512)
    raise FileNotFoundError(sid)


def backfill(pred, baseline, mask, r=8):
    b = torch.from_numpy((mask > 0.5).astype(np.float32))[None, None]
    dil = F.max_pool2d(b, kernel_size=2 * r + 1, stride=1, padding=r)[0, 0].numpy()
    return (1.0 - dil) * pred + dil * baseline


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = FINDNet(FINDNetArgs()).to(dev)
    net.load_state_dict(torch.load(f"{D}/findnet_stage1_v2.1.pt", map_location=dev,
                                   weights_only=False)["find_net"], assign=True)
    net.eval()
    t = lambda a: torch.from_numpy(a[None, None].astype(np.float32)).to(dev)

    print(f"{'size':7s} {'state':8s} {'p50':>6s} {'p90':>6s} {'p99':>7s} {'max':>7s} "
          f"| {'<50':>6s} {'50-100':>7s} {'100-200':>8s} {'200-400':>8s} {'>400':>6s}")
    rows = {}
    for size, sid in PICKS.items():
        gt = rd(sid, "Target"); bl = rd(sid, "Baseline")
        li = rd(sid, "LI"); mk = rd(sid, "Mask")
        with torch.no_grad():
            raw = net.reconstruct(t(bl), t(li), t(mk))[0, 0].cpu().numpy()
        fn = backfill(raw, bl, mk)
        sel = tissue_mask(mk, gt, r=8)
        states = {"after": bl, "FN": fn}
        cf_p = os.path.join(CACHE, f"{sid}_cleanFBP.npy")
        if os.path.exists(cf_p):
            states["before"] = np.load(cf_p).astype(np.float32)
        for name, img in states.items():
            e = np.abs(img - gt)[sel]
            q = np.percentile(e, [50, 90, 99])
            hist = np.histogram(e, bins=EDGES)[0] / e.size
            rows.setdefault(size, {})[name] = (q, float(e.max()), hist, float(e.mean()))
            print(f"{size:7s} {name:8s} {q[0]:6.1f} {q[1]:6.1f} {q[2]:7.1f} {e.max():7.0f} "
                  f"| " + " ".join(f"{h:6.3f}" for h in hist))

    # 图：每个尺寸的误差分布（阶梯 CDF）
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "font.size": 9})
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
    colors = {"before": "#999999", "after": "#d62728", "FN": "#1f77b4"}
    for ax, size in zip(axes, PICKS):
        for name in ("before", "after", "FN"):
            if name not in rows[size]:
                continue
            hist = rows[size][name][2]
            # 用累积作为近似 CDF（按箱右界）
            cum = np.cumsum(hist)
            xs = [0, 50, 100, 200, 400, 800]
            ax.plot(xs, [0] + list(cum), "o-", color=colors[name], lw=1.6, ms=4, label=name)
        ax.set_xscale("log")
        ax.set_xlabel("|error| (HU)"); ax.set_title(size, fontsize=10)
        ax.grid(color="#d9d9d9", lw=0.5, alpha=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("cumulative fraction of tissue")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Error CDF per state (bins: <50 / 50-100 / 100-200 / 200-400 / >400 HU)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(OUT, "error_cdf.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[save]", os.path.join(OUT, "error_cdf.png"))


if __name__ == "__main__":
    main()
