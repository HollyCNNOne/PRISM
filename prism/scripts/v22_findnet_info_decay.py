"""v22_findnet_info_decay.py — FIND-Net 性能 vs 单切片可观测信息（V2.2-baseline-FIND-NET-1 的 P2 重算）。

与旧版 `v22_prior_gain.py` Part2 的区别：**三条曲线统一评分域**。

    旧版（作废）
        before / after  ← metal_sweep.csv，域 = (mk ≤ 1e-3) & (gt > −500)   含 8px 回填环
        fn              ← tissue_mask                                       排回填环
    → 三条曲线在两个域上算，不可比。

    本脚本
        before / after  ← metal_sweep_tm.csv（域 = tissue_mask）
        fn              ← tissue_mask
    → 同域。

输出：逐片 CSV（金属量 / 三曲线误差 / 信息缺失）/ 分箱中位图 / 饱和拟合。

变量说明
    横轴 metal          = mask.sum()，单切片金属量（体素当量）
    可观测信息代理       = 1 − ΔInfoLoss@100（越大表示单切片上可观测的信息越多）
"""
import argparse, csv, glob, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

_FR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _FR)
from prism.eval.domain import tissue_mask
from prism.models.find_net import FINDNet, FINDNetArgs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MASK_EPS = 1e-3
DR = 3000.0


def rd(p):
    return np.fromfile(p, np.float32).reshape(512, 512)


def find_split(d, sid, sub):
    for sp in ("train", "val"):
        p = os.path.join(d, sp, sub, f"{sid}.raw")
        if os.path.exists(p):
            return p
    return None


def backfill(pred, baseline, mask, r=8):
    b = torch.from_numpy((mask > 0.5).astype(np.float32))[None, None]
    dil = F.max_pool2d(b, kernel_size=2 * r + 1, stride=1, padding=r)[0, 0].numpy()
    return (1.0 - dil) * pred + dil * baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--ckpt", default="/root/autodl-tmp/PRISM/training_data_astra/findnet_stage1_v2.1.pt")
    ap.add_argument("--sweep", default="/root/autodl-tmp/PRISM/v2.2/metal_tm/metal_sweep_tm.csv")
    ap.add_argument("--outdir", default="/root/autodl-tmp/PRISM/v2.2/findnet_decay")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    dev = args.device

    net = FINDNet(FINDNetArgs()).to(dev)
    net.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False)["find_net"],
                        assign=True)
    net.eval()
    t = lambda a: torch.from_numpy(a[None, None].astype(np.float32)).to(dev)

    rows_in = list(csv.DictReader(open(args.sweep)))
    print(f"[data] {len(rows_in)} 片（域 = tissue_mask）", flush=True)

    pts = []
    for i, r in enumerate(rows_in):
        sid = r["sid"]
        gt = rd(find_split(args.data_dir, sid, "Target"))
        bl = rd(find_split(args.data_dir, sid, "Baseline"))
        li = rd(find_split(args.data_dir, sid, "LI"))
        mk = rd(find_split(args.data_dir, sid, "Mask"))
        with torch.no_grad():
            raw = net.reconstruct(t(bl), t(li), t(mk))[0, 0].cpu().numpy()
        fn = backfill(raw, bl, mk)
        sel = tissue_mask(mk, gt, r=8)
        e_f = np.abs(fn - gt)[sel]
        pts.append(dict(sid=sid, cat=r["cat"], metal=float(r["metal"]),
                        before=float(r["mae_before"]), after=float(r["mae_after"]),
                        fn=float(e_f.mean()),
                        d100=float(r["d100"]), f100_before=float(r["f100_before"]),
                        f100_after=float(r["f100_after"])))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(rows_in)}", flush=True)

    out_csv = os.path.join(args.outdir, "findnet_decay.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pts[0].keys()))
        w.writeheader(); w.writerows(pts)
    print("[save]", out_csv)

    x = np.array([p["metal"] for p in pts])
    yb = np.array([p["before"] for p in pts])
    ya = np.array([p["after"] for p in pts])
    yf = np.array([p["fn"] for p in pts])

    print("\n=== 三条曲线（同一评分域 tissue_mask）===")
    print(f"{'metal分箱中位':>14s} {'植入前地板':>10s} {'Baseline':>10s} {'FIND-Net':>10s} {'FN/地板':>9s}")
    qs = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
    binned = []
    for a, b in zip(qs[:-1], qs[1:]):
        m = (x >= a) & (x <= b)
        if m.sum() >= 3:
            row = (float(np.median(x[m])), float(np.median(yb[m])),
                   float(np.median(ya[m])), float(np.median(yf[m])))
            binned.append(row)
            print(f"{row[0]:14.1f} {row[1]:10.1f} {row[2]:10.1f} {row[3]:10.1f} "
                  f"{row[3]/max(row[1],1e-9):9.3f}")

    print("\n=== 全局 ===")
    print(f"  floor  中位 {np.median(yb):.1f} | Baseline 中位 {np.median(ya):.1f} | "
          f"FIND-Net 中位 {np.median(yf):.1f}")
    print(f"  FIND-Net < floor 的比例: {(yf < yb).mean()*100:.1f}%  ({int((yf<yb).sum())}/{len(yf)})")
    print(f"  中位比值 floor/FN = {np.median(yb)/np.median(yf):.3f}")

    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.scatter(x, yb, s=10, c="#999999", alpha=0.45, label="before implant (floor)")
    ax.scatter(x, ya, s=10, c="#d62728", alpha=0.30, label="Baseline (artifact)")
    ax.scatter(x, yf, s=12, c="#1f77b4", alpha=0.60, label="FIND-Net (unified domain)")
    for arr, c in [(yb, "#666666"), (ya, "#d62728"), (yf, "#1f77b4")]:
        cx = [r[0] for r in binned]
        cy = {"#666666": [r[1] for r in binned], "#d62728": [r[2] for r in binned],
              "#1f77b4": [r[3] for r in binned]}[c]
        ax.plot(cx, cy, "-", color=c, lw=2.0)
    ax.set_xscale("log")
    ax.set_xlabel("metal amount (voxel-equivalent, log)")
    ax.set_ylabel("tissue MAE (HU)")
    ax.grid(color="#d9d9d9", lw=0.5, alpha=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("FIND-Net vs single-slice observable information "
                 "(all three curves on tissue_mask)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "findnet_decay.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[save]", os.path.join(args.outdir, "findnet_decay.png"))


if __name__ == "__main__":
    main()
