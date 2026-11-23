"""v22_metal_sweep.py — 信息缺失 vs 金属量（连续横轴，饱和曲线）。

对**所有含金属切片**（mask.max>1e-3）计算：
    金属量  = mask.sum()（体素当量）
    植入后  = Baseline 相对 GT 的组织 MAE / >50HU / >100HU
    植入前  = 零金属 FBP（同 seed/同管线）相对 GT 的同口径指标
    ΔInfoLoss(τ) = frac_after(τ) − frac_before(τ)     ← 伪影增量

输出：CSV + 散点图（按 size 上色）+ 分箱中位线 + 饱和拟合 y = ymax·x/(K+x)
"""
import argparse, glob, json, os, sys
from multiprocessing import Pool

import numpy as np
from skimage.transform import iradon

sys.path.insert(0, "/root/autodl-tmp/PRISM/3D_Coil")
from parallel_beam import (simulate_sino, THETA, PIXEL_MM, BASELINE_FILTER,
                           LO, HI, mu_to_hu)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/root/autodl-tmp/PRISM/training_data_astra"
OUT = "/root/autodl-tmp/PRISM/v2.2/metal"
MASK_EPS = 1e-3
AIR_HU = -500.0


def rd(path):
    return np.fromfile(path, np.float32).reshape(512, 512)


def find(data_dir, sid, sub):
    for sp in ("train", "val"):
        p = os.path.join(data_dir, sp, sub, f"{sid}.raw")
        if os.path.exists(p):
            return p
    return None


def one(job):
    sid, cat, data_dir = job
    try:
        gt = rd(find(data_dir, sid, "Target"))
        bl = rd(find(data_dir, sid, "Baseline"))
        mk = rd(find(data_dir, sid, "Mask"))
        metal = float(mk.sum())
        if metal <= 0.5:
            return None
        seed = int(sid.rsplit("_", 1)[1])
        # 植入前：零金属同管线重建
        L = simulate_sino(gt, np.zeros_like(gt), cat, seed)
        mu = iradon(L / PIXEL_MM, theta=THETA, filter_name=BASELINE_FILTER, circle=True)
        cf = np.clip(mu_to_hu(mu), LO, HI).astype(np.float32)
        cf[gt <= AIR_HU] = -1000.0

        sel = (mk <= MASK_EPS) & (gt > AIR_HU)
        if sel.sum() < 1000:
            return None
        e_a = np.abs(bl - gt)[sel]
        e_b = np.abs(cf - gt)[sel]
        return dict(sid=sid, cat=cat, metal=metal,
                    mae_after=float(e_a.mean()), mae_before=float(e_b.mean()),
                    f50_after=float((e_a > 50).mean()), f50_before=float((e_b > 50).mean()),
                    f100_after=float((e_a > 100).mean()), f100_before=float((e_b > 100).mean()))
    except Exception as e:
        print("[fail]", sid, type(e).__name__, e, flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=D)
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--nproc", type=int, default=14)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    metas = [json.load(open(f)) for f in glob.glob(os.path.join(args.data_dir, "meta", "*.json"))]
    jobs = []
    for m in metas:
        sid = m["sample_id"]
        mp = find(args.data_dir, sid, "Mask")
        if not mp:
            continue
        if rd(mp).max() <= MASK_EPS:
            continue
        jobs.append((sid, m.get("category", "medium"), args.data_dir))
    print(f"[data] 含金属切片 {len(jobs)} 张", flush=True)

    with Pool(args.nproc) as pool:
        rows = [r for r in pool.imap_unordered(one, jobs) if r]
    print(f"[done] 有效 {len(rows)} 张", flush=True)

    for r in rows:
        r["d50"] = r["f50_after"] - r["f50_before"]
        r["d100"] = r["f100_after"] - r["f100_before"]

    import csv
    csv_path = os.path.join(args.outdir, "metal_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("[save]", csv_path)

    # ---- 画图 ----
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "font.size": 10, "axes.linewidth": 0.8})
    colors = {"small": "#8ecae6", "medium": "#219ebc", "large": "#fb8500", "giant": "#9d0208"}
    x = np.array([r["metal"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    for ax, ykey, ylab, tau in [(axes[0], "d100", r"$\Delta$InfoLoss  ($\tau$=100 HU)", 100),
                                (axes[1], "mae_after", "Baseline MAE (HU)", None)]:
        y = np.array([r[ykey] for r in rows])
        for cat, c in colors.items():
            m = np.array([r["cat"] == cat for r in rows])
            if m.any():
                ax.scatter(x[m], y[m], s=14, c=c, alpha=0.6,
                           edgecolors="none", label=cat)
        # 分箱中位线
        qs = np.quantile(x, np.linspace(0, 1, 9))
        qs = np.unique(qs)
        cx, cy = [], []
        for a, b in zip(qs[:-1], qs[1:]):
            m = (x >= a) & (x <= b)
            if m.sum() >= 3:
                cx.append(np.median(x[m])); cy.append(np.median(y[m]))
        ax.plot(cx, cy, "-", color="#333333", lw=2.0, label="binned median")
        # 饱和拟合 y = ymax·x/(K+x)
        try:
            from scipy.optimize import curve_fit
            f = lambda xx, ymax, K: ymax * xx / (K + xx)
            p, _ = curve_fit(f, x, y, p0=[y.max(), np.median(x)], maxfev=20000)
            xs = np.linspace(x.min(), x.max(), 200)
            ax.plot(xs, f(xs, *p), "--", color="#1f77b4", lw=1.6,
                    label=f"saturation fit  $y_{{max}}$={p[0]:.3g}, K={p[1]:.3g}")
        except Exception as e:
            print("[fit] skip", e)
        ax.set_xscale("log")
        ax.set_xlabel("metal amount (voxel-equivalent, log)")
        ax.set_ylabel(ylab)
        ax.grid(color="#d9d9d9", lw=0.5, alpha=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Information loss vs metal amount  (N={len(rows)} metal-bearing slices)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    png = os.path.join(args.outdir, "metal_sweep.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[save]", png)


if __name__ == "__main__":
    main()
