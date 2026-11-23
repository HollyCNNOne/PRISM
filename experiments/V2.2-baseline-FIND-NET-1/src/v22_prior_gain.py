"""v22_prior_gain.py — 先验增益实验（两部分）。

Part 1｜无金属切片：FIND-Net 能否胜过「干净重建」本身？
    对无金属切片（mask.max≈0），Baseline 就是零金属 FBP。
    比较 |Baseline−GT| 与 |FIND-Net−GT|，并出误差热力图。

Part 2｜分析 A：FIND-Net 误差 vs 单切片金属量
    叠加三条曲线：植入前地板（clean FBP）、植入后（Baseline）、FIND-Net。
    找「先验不再能替代观测」的交叉点。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from prism.eval.domain import tissue_mask
import argparse, csv, glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/prism_framework")
from prism.models.find_net import FINDNet, FINDNetArgs
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/root/autodl-tmp/PRISM/training_data_astra"
OUT = "/root/autodl-tmp/PRISM/v2.2/prior_gain"
MASK_EPS = 1e-3
AIR = -500.0


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
    ap.add_argument("--ckpt", default=f"{D}/findnet_stage1_v2.1.pt")
    ap.add_argument("--patient", default="patients_head2")
    ap.add_argument("--n-clean", type=int, default=40)
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    dev = args.device
    net = FINDNet(FINDNetArgs()).to(dev)
    net.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False)["find_net"], assign=True)
    net.eval()
    t = lambda a: torch.from_numpy(a[None, None].astype(np.float32)).to(dev)

    meta = {}
    for f in glob.glob(os.path.join(D, "meta", "*.json")):
        m = json.load(open(f)); meta[m["sample_id"]] = m
    held = [s for s, m in meta.items() if m.get("patient_id") == args.patient]
    held.sort()

    # ---------- Part 1：无金属切片 ----------
    clean_ids = []
    for sid in held:
        p = find_split(D, sid, "Mask")
        if p and rd(p).max() <= MASK_EPS:
            clean_ids.append(sid)
        if len(clean_ids) >= args.n_clean:
            break
    print(f"[Part1] 无金属切片 {len(clean_ids)} 张", flush=True)

    recs, maps = [], []
    for sid in clean_ids:
        gt = rd(find_split(D, sid, "Target")); bl = rd(find_split(D, sid, "Baseline"))
        li = rd(find_split(D, sid, "LI")); mk = rd(find_split(D, sid, "Mask"))
        with torch.no_grad():
            fn = net.reconstruct(t(bl), t(li), t(mk))[0, 0].cpu().numpy()
        sel = tissue_mask(mk, gt, r=8)
        e_b = np.abs(bl - gt); e_f = np.abs(fn - gt)
        recs.append((sid, float(e_b[sel].mean()), float(e_f[sel].mean())))
        maps.append((sid, gt, bl, fn, e_b, e_f, sel))

    mb = np.mean([r[1] for r in recs]); mf = np.mean([r[2] for r in recs])
    print(f"[Part1] 平均 MAE: 干净重建 {mb:.1f} → FIND-Net {mf:.1f} HU "
          f"({mb/max(mf,1e-6):.2f}× 改善)", flush=True)
    print(f"[Part1] 胜率: {np.mean([r[2] < r[1] for r in recs])*100:.0f}% 的切片 FIND-Net 更好")

    # 热力图：挑「改善量位于中位」的那张
    recs_sorted = sorted(recs, key=lambda r: r[1] - r[2])
    mid_sid = recs_sorted[len(recs_sorted) // 2][0]
    m = next(x for x in maps if x[0] == mid_sid)
    sid, gt, bl, fn, e_b, e_f, sel = m

    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "font.size": 9})
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.1))
    panels = [(gt, "GT", "gray", -200, 400), (bl, "Clean recon (no metal)", "gray", -200, 400),
              (fn, "FIND-Net output", "gray", -200, 400),
              (np.where(sel, e_b, np.nan), "|重建 − GT|", "inferno", 0, 200),
              (np.where(sel, e_f, np.nan), "|FIND-Net − GT|", "inferno", 0, 200)]
    for ax, (img, tt, cm, lo, hi) in zip(axes, panels):
        im = ax.imshow(img, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(tt, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"Metal-free slice {sid}: FIND-Net also removes reconstruction noise/blur  "
                 f"({mb:.1f} -> {mf:.1f} HU, mean of {len(recs)} slices)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(args.outdir, "prior_gain_clean.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 逐片散点：重建 vs FIND-Net
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    xs = [r[1] for r in recs]; ys = [r[2] for r in recs]
    ax.scatter(xs, ys, s=22, alpha=0.75, c="#1f77b4", edgecolors="none")
    lim = [0, max(xs + ys) * 1.08]
    ax.plot(lim, lim, "--", color="#999", lw=1, label="y = x")
    ax.set_xlabel("MAE of clean reconstruction (HU)")
    ax.set_ylabel("MAE of FIND-Net (HU)")
    ax.set_title(f"Metal-free slices (n={len(recs)})", fontsize=10)
    ax.grid(color="#d9d9d9", lw=0.5, alpha=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "prior_gain_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- Part 2：分析 A，交叉点 ----------
    sweep = os.path.join("/root/autodl-tmp/PRISM/v2.2/metal/metal_sweep.csv")
    if not os.path.exists(sweep):
        print("[Part2] 缺 metal_sweep.csv，跳过"); return
    rows = list(csv.DictReader(open(sweep)))
    pts = []
    for r in rows:
        sid = r["sid"]
        gt = rd(find_split(D, sid, "Target")); bl = rd(find_split(D, sid, "Baseline"))
        li = rd(find_split(D, sid, "LI")); mk = rd(find_split(D, sid, "Mask"))
        with torch.no_grad():
            raw = net.reconstruct(t(bl), t(li), t(mk))[0, 0].cpu().numpy()
        fn = backfill(raw, bl, mk)
        sel = tissue_mask(mk, gt, r=8)
        pts.append(dict(metal=float(r["metal"]), before=float(r["mae_before"]),
                        after=float(r["mae_after"]), fn=float(np.abs(fn - gt)[sel].mean())))
    print(f"[Part2] {len(pts)} 片", flush=True)

    x = np.array([p["metal"] for p in pts])
    yb = np.array([p["before"] for p in pts])
    ya = np.array([p["after"] for p in pts])
    yf = np.array([p["fn"] for p in pts])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.scatter(x, yb, s=10, c="#999999", alpha=0.45, label="before implant (floor)")
    ax.scatter(x, ya, s=10, c="#d62728", alpha=0.35, label="Baseline (artifact)")
    ax.scatter(x, yf, s=12, c="#1f77b4", alpha=0.6, label="FIND-Net (+backfill)")
    qs = np.unique(np.quantile(x, np.linspace(0, 1, 10)))
    for arr, c in [(yb, "#999999"), (ya, "#d62728"), (yf, "#1f77b4")]:
        cx, cy = [], []
        for a, b in zip(qs[:-1], qs[1:]):
            msk = (x >= a) & (x <= b)
            if msk.sum() >= 3:
                cx.append(np.median(x[msk])); cy.append(np.median(arr[msk]))
        ax.plot(cx, cy, "-", color=c, lw=2.0)
    ax.set_xscale("log")
    ax.set_xlabel("metal amount (voxel-equivalent, log)")
    ax.set_ylabel("tissue MAE (HU)")
    ax.grid(color="#d9d9d9", lw=0.5, alpha=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Prior gain vs metal amount: FIND-Net beats the floor only at low metal", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "prior_gain_vs_metal.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 交叉点：分箱后 FN 首次超过 floor
    cross = None
    for a, b in zip(qs[:-1], qs[1:]):
        msk = (x >= a) & (x <= b)
        if msk.sum() >= 3 and np.median(yf[msk]) > np.median(yb[msk]):
            cross = float(np.median(x[msk])); break
    print(f"[Part2] 交叉点（FN 首次差于植入前地板）≈ 金属量 {cross}", flush=True)
    print("[save] prior_gain_clean.png / prior_gain_scatter.png / prior_gain_vs_metal.png")


if __name__ == "__main__":
    main()
