"""v22_error_dist_multi.py — 误差分布（V2.2-baseline-FIND-NET-2）：多切片版。

与旧版 `v22_error_dist.py` 的区别
    旧版：每个尺寸硬编码 1 张切片（PICKS），n=1。
    本版：**每条序列取一张「代表片」**——金属量最大的那一片。

为什么取金属量最大的那一片
    线圈是三维体，沿 z 的横截面是「小 → 标称尺寸 → 小」。
    序列中金属量最大的切片才代表该植入体的标称 size；其余切片只切到边缘。

输出（每张切片三状态）
    before : 零金属同管线 FBP（simulate_sino(gt, 0, cat, seed) → iradon）
    after  : Baseline
    FN     : FIND-Net + 金属回填
    指标：p50/p90/p99/max + 分箱占比（<50 / 50-100 / 100-200 / 200-400 / >400）
    评分域：tissue_mask(mk, gt, r=8)
"""
import argparse, csv, glob, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

_FR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _FR)
from prism.eval.domain import tissue_mask
from prism.models.find_net import FINDNet, FINDNetArgs

sys.path.insert(0, "/root/autodl-tmp/PRISM/3D_Coil")
from parallel_beam import (simulate_sino, THETA, PIXEL_MM, BASELINE_FILTER, LO, HI, mu_to_hu)
from skimage.transform import iradon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MASK_EPS = 1e-3
AIR = -500.0
EDGES = [0, 50, 100, 200, 400, 1e9]
LABELS = ["<50", "50-100", "100-200", "200-400", ">400"]


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


def clean_fbp(gt, cat, seed):
    L = simulate_sino(gt, np.zeros_like(gt), cat, seed)
    mu = iradon(L / PIXEL_MM, theta=THETA, filter_name=BASELINE_FILTER, circle=True)
    cf = np.clip(mu_to_hu(mu), LO, HI).astype(np.float32)
    cf[gt <= AIR] = -1000.0
    return cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--ckpt", default="/root/autodl-tmp/PRISM/training_data_astra/findnet_stage1_v2.1.pt")
    ap.add_argument("--patient", default="patients_head2")
    ap.add_argument("--outdir", default="/root/autodl-tmp/PRISM/v2.2/err_dist_multi")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    dev = args.device

    net = FINDNet(FINDNetArgs()).to(dev)
    net.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False)["find_net"],
                        assign=True)
    net.eval()
    t = lambda a: torch.from_numpy(a[None, None].astype(np.float32)).to(dev)

    meta = [json.load(open(f)) for f in glob.glob(os.path.join(args.data_dir, "meta", "*.json"))]
    by_seq = {}
    for m in meta:
        if m.get("patient_id") != args.patient:
            continue
        by_seq.setdefault(m["sequence_id"], []).append(m)

    picks = []
    for seq, ms in by_seq.items():
        best, bestm = None, -1.0
        for m in ms:
            p = find_split(args.data_dir, m["sample_id"], "Mask")
            if not p:
                continue
            v = float((rd(p) > MASK_EPS).sum())
            if v > bestm:
                best, bestm = m, v
        if best is None or bestm <= 0.5:
            continue
        picks.append((best["category"], best["sample_id"], bestm, len(ms)))
    picks.sort(key=lambda x: (x[0], x[1]))
    print(f"[pick] {len(picks)} 条序列的代表片（金属量最大）", flush=True)
    for c, s, v, z in picks:
        print(f"  {c:7s} {s} 金属量={v:8.1f} 序列长={z}", flush=True)

    rows = []
    for cat, sid, mval, zlen in picks:
        gt = rd(find_split(args.data_dir, sid, "Target"))
        bl = rd(find_split(args.data_dir, sid, "Baseline"))
        li = rd(find_split(args.data_dir, sid, "LI"))
        mk = rd(find_split(args.data_dir, sid, "Mask"))
        cf = clean_fbp(gt, cat, int(sid.rsplit("_", 1)[1]))
        with torch.no_grad():
            raw = net.reconstruct(t(bl), t(li), t(mk))[0, 0].cpu().numpy()
        fn = backfill(raw, bl, mk)
        sel = tissue_mask(mk, gt, r=8)
        for name, img in (("before", cf), ("after", bl), ("FN", fn)):
            e = np.abs(img - gt)[sel]
            q = np.percentile(e, [50, 90, 99])
            h = np.histogram(e, bins=EDGES)[0] / e.size
            rows.append(dict(size=cat, sid=sid, metal=mval, seqlen=zlen, state=name,
                             p50=q[0], p90=q[1], p99=q[2], max=float(e.max()),
                             mae=float(e.mean()),
                             **{f"b{i}": float(h[i]) for i in range(len(LABELS))}))
        print(f"  done {sid}", flush=True)

    out = os.path.join(args.outdir, "err_dist_multi.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("[save]", out)

    # ---- 汇总：每尺寸跨代表片取中位 ----
    sizes = ["small", "medium", "large", "giant"]
    print("\n=== 每尺寸（跨代表片取中位）===")
    print(f"{'size':7s} {'state':7s} {'n':>2s} {'p50':>7s} {'p90':>8s} {'p99':>9s} {'max':>7s} {'>400':>7s}")
    summ = {}
    for s in sizes:
        for st in ("before", "after", "FN"):
            v = [r for r in rows if r["size"] == s and r["state"] == st]
            if not v:
                continue
            med = {k: float(np.median([x[k] for x in v])) for k in ("p50", "p90", "p99", "max", "mae", "b4")}
            summ[(s, st)] = med
            print(f"{s:7s} {st:7s} {len(v):2d} {med['p50']:7.1f} {med['p90']:8.1f} "
                  f"{med['p99']:9.1f} {med['max']:7.0f} {med['b4']*100:6.2f}%")
        print()

    print("=== FN / before 的比值（<1 = FIND-Net 更好）===")
    print(f"{'size':7s} {'p50':>7s} {'p90':>7s} {'p99':>7s}")
    for s in sizes:
        if (s, "FN") not in summ:
            continue
        a, b = summ[(s, "before")], summ[(s, "FN")]
        print(f"{s:7s} {b['p50']/a['p50']:7.3f} {b['p90']/a['p90']:7.3f} {b['p99']/a['p99']:7.3f}")

    # ---- 图：四尺寸 CDF ----
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "font.size": 9})
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
    colors = {"before": "#999999", "after": "#d62728", "FN": "#1f77b4"}
    names = {"before": "before implant (clean FBP)", "after": "Baseline (artifact)",
             "FN": "FIND-Net"}
    for ax, s in zip(axes, sizes):
        for st in ("before", "after", "FN"):
            v = [r for r in rows if r["size"] == s and r["state"] == st]
            if not v:
                continue
            # 用分箱占比画阶梯 CDF（跨片平均）
            cum = np.zeros(len(LABELS))
            for i in range(len(LABELS)):
                cum[i] = np.mean([x[f"b{i}"] for x in v])
            ys = np.concatenate([[0.0], np.cumsum(cum)])
            xs = np.array([EDGES[0], EDGES[1], EDGES[2], EDGES[3], EDGES[4], 4000.0])
            ax.step(xs, ys, where="post", color=colors[st], lw=1.6, label=names[st])
        ax.set_xscale("log"); ax.set_xlim(10, 4000); ax.set_ylim(0, 1.02)
        ax.set_title(f"{s}", fontsize=10)
        ax.set_xlabel("|error| (HU, log)")
        ax.grid(color="#d9d9d9", lw=0.5, alpha=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("cumulative fraction of tissue voxels")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Error distribution — representative slice per sequence (n per size varies), "
                 f"held-out {args.patient}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(args.outdir, "err_dist_multi.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[save]", os.path.join(args.outdir, "err_dist_multi.png"))


if __name__ == "__main__":
    main()
