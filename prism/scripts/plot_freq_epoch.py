"""plot_freq_epoch.py — 画 3 频带 z 轴占比随 epoch/steps 变化的训练曲线。

读 FreqRecorder 的 CSV（epoch,pillar,band,gt,out,err），对每个 band 跨 pillar 平均，
画 epoch(x) vs 占比(y) 三条曲线（低=蓝/中=红/高=黄），GT 占比画水平参考虚线。
论文风：衬线字体、浅横向网格、无多余装饰。
"""
import argparse, os, csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BANDS = ["low", "mid", "high"]
COLORS = {"low": "#1f77b4", "mid": "#d62728", "high": "#f5c542"}
LABELS = {"low": "Low freq", "mid": "Mid freq", "high": "High freq"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--xlabel", default="epoch")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "font.size": 11, "axes.linewidth": 0.8,
    })
    rows = list(csv.DictReader(open(args.csv)))
    by_band = defaultdict(lambda: defaultdict(list))
    gt_by_band = defaultdict(list)
    for r in rows:
        e = int(r["epoch"]); b = r["band"]; out = float(r["out"]); gt = float(r["gt"])
        by_band[b][e].append(out)
        gt_by_band[b].append(gt)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for b in BANDS:
        if b not in by_band:
            continue
        xs = sorted(by_band[b])
        ys = [float(np.mean(by_band[b][e])) for e in xs]
        # 平滑（小窗口移动平均，保留训练波动）
        if len(ys) >= 5:
            k = 3
            kern = np.ones(k) / k
            ys = np.convolve(ys, kern, mode="same")
            ys[:k//2] = by_band[b][xs[0]][0]
        ax.plot(xs, ys, color=COLORS[b], lw=1.6, label=LABELS[b])
        g = float(np.mean(gt_by_band[b]))
        ax.axhline(g, color=COLORS[b], ls="--", lw=0.9, alpha=0.55)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel("z-axis frequency proportion")
    ax.grid(axis="y", color="#d9d9d9", lw=0.6, alpha=0.7)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10)
    if args.title:
        ax.set_title(args.title, fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
