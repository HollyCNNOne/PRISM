"""v22_info_loss.py — 不同 size 线圈的信息缺失度分析。

对 small/medium/large/giant 各选一个典型切片（金属 mask 最大的切片 = 线圈中心），
排除金属植入区域后，用 d_j 光子充分度算非金属组织的观测分区：
    完全阴影 r<1/9、部分阴影 1/9<=r<=9、充分 r>9（r=d_j/d_ref）。
信息缺失度 = 1 - 充分区占比 =（完全阴影 + 部分阴影）/ 非金属组织。
输出 v2.2/info_loss/。
"""
import argparse, glob, json, os, sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIZES = ["small", "medium", "large", "giant"]


def read_raw(path):
    return np.fromfile(path, np.float32).reshape(512, 512)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--outdir", default="/root/autodl-tmp/PRISM/v2.2/info_loss")
    args = ap.parse_args()

    meta_dir = os.path.join(args.data_dir, "meta")
    # 按 aneurysm size (category) 分组，记录每个 sample 的 mask 金属量
    by_size = {s: [] for s in SIZES}
    for f in glob.glob(os.path.join(meta_dir, "*.json")):
        m = json.load(open(f))
        cat = m.get("category")  # aneurysm size: small/medium/large/giant
        if cat not in SIZES:
            continue
        sid = m["sample_id"]
        by_size[cat].append(sid)

    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    for size in SIZES:
        sids = by_size[size]
        if not sids:
            print(f"[skip] {size} 无样本")
            continue
        # 选典型切片：mask 金属量最大的切片（线圈中心，伪影最强）
        best_sid, best_m = None, -1
        for sid in sids:
            mp = None
            for sp in ("train", "val"):
                p = os.path.join(args.data_dir, sp, "Mask", f"{sid}.raw")
                if os.path.exists(p):
                    mp = p
                    break
            if mp is None:
                continue
            m = read_raw(mp)
            s = float((m > 0.5).sum())
            if s > best_m:
                best_m = s
                best_sid = sid
        # 读 d_j 与 mask
        dj = None
        mask = None
        for sp in ("train", "val"):
            dp = os.path.join(args.data_dir, sp, "Phys", f"{best_sid}_d_j.npy")
            mp = os.path.join(args.data_dir, sp, "Mask", f"{best_sid}.raw")
            if os.path.exists(dp) and os.path.exists(mp):
                dj = np.load(dp).astype(np.float32)[0]
                mask = read_raw(mp)
                break
        if dj is None:
            print(f"[skip] {size} 无 d_j")
            continue
        nonmetal = mask < 0.5
        d_ref = float(np.median(dj[nonmetal]))
        if d_ref <= 0:
            d_ref = 1.0
        r = dj / d_ref
        complete = nonmetal & (r < 1.0 / 9.0)
        partial = nonmetal & (r >= 1.0 / 9.0) & (r <= 9.0)
        suff = nonmetal & (r > 9.0)
        n = int(nonmetal.sum())
        frac = {"complete": complete.sum() / n, "partial": partial.sum() / n,
                "sufficient": suff.sum() / n}
        info_loss = frac["complete"] + frac["partial"]
        rows.append((size, frac, info_loss, best_sid))
        print(f"[{size}] sid={best_sid} d_ref={d_ref:.1f} "
              f"充分={frac['sufficient']:.3f} 部分阴影={frac['partial']:.3f} "
              f"完全阴影={frac['complete']:.3f} 信息缺失度={info_loss:.3f}")

        # 每个 size 存一张 d_j 阴影可视化（非金属 + 金属标注）
        fig, ax = plt.subplots(figsize=(4.5, 4))
        disp = np.where(nonmetal, np.log1p(np.clip(dj, 0, None)), np.nan)
        im = ax.imshow(disp, cmap="viridis", vmin=0, vmax=np.nanpercentile(disp, 99))
        ax.imshow(np.where(mask > 0.5, 1.0, np.nan), cmap="Reds", alpha=0.7)
        ax.set_title(f"{size} — d_j shadow (red=metal)")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"shadow_{size}.png"), dpi=120)
        plt.close(fig)

    # 曲线：size vs 信息缺失度（不是柱状图）
    if rows:
        plt.rcParams.update({"font.family": "serif",
                             "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                             "font.size": 11, "axes.linewidth": 0.8})
        xs = np.arange(len(rows))
        ys = [r[2] for r in rows]
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        ax.fill_between(xs, 0, ys, color="#d62728", alpha=0.3)
        ax.plot(xs, ys, color="#d62728", lw=1.6, marker="o", ms=5)
        ax.set_xticks(xs)
        ax.set_xticklabels([r[0] for r in rows])
        ax.set_ylabel("information loss (shadow fraction)")
        ax.set_xlabel("coil size")
        ax.grid(axis="y", color="#d9d9d9", lw=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title("metal-artifact information loss vs coil size (non-metal)", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "info_loss_vs_size.png"), dpi=150)
        plt.close(fig)

    with open(os.path.join(args.outdir, "info_loss.csv"), "w") as f:
        f.write("size,sufficient,partial,complete,info_loss\n")
        for size, frac, info_loss, sid in rows:
            f.write(f"{size},{frac['sufficient']:.6f},{frac['partial']:.6f},{frac['complete']:.6f},{info_loss:.6f}\n")
    print("[done]")


if __name__ == "__main__":
    main()
