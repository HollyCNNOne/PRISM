"""v22_obs_info_loss.py — 观测端信息缺失度（植入前 vs 植入后，图像域）。

观测 = Baseline（重建带伪影）；真值 = GT。
    Err_before(x,y) = |FBP_clean(x,y) − GT(x,y)|     零金属重建（只有重建模糊）
    Err_after (x,y) = |Baseline(x,y)  − GT(x,y)|     模糊 + 金属伪影（条纹 + 阴影）

信息缺失度 = 非金属组织里 Err 超过 τ 的比例；τ 给固定 HU 阈值，并画生存曲线
（避免单一阈值的任意性）。伪影造成的额外缺失 = after − before。

只用**有金属的**典型切片（每 size 取金属体素当量最大的一张）。
"""
import argparse, glob, json, os, sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIZES = ["small", "medium", "large", "giant"]
COIL = "/root/autodl-tmp/PRISM/3D_Coil"
MASK_EPS = 1e-3
AIR_HU = -500.0


def read_raw(p):
    return np.fromfile(p, np.float32).reshape(512, 512)


def find_split(data_dir, sid, sub):
    for sp in ("train", "val"):
        p = os.path.join(data_dir, sp, sub, f"{sid}.raw")
        if os.path.exists(p):
            return p
    return None


def clean_fbp(clean, cat, sid, cache_dir):
    """零金属 FBP（植入前观测）——用解析线唯一真源 parallel_beam，与 Baseline 同噪声实现。"""
    os.makedirs(cache_dir, exist_ok=True)
    cpath = os.path.join(cache_dir, f"{sid}_cleanFBP.npy")
    if os.path.exists(cpath):
        return np.load(cpath).astype(np.float32)

    sys.path.insert(0, COIL)
    from parallel_beam import (simulate_sino, THETA, PIXEL_MM, BASELINE_FILTER,
                               LO, HI, mu_to_hu)
    from skimage.transform import iradon

    seed = int(str(sid).rsplit("_", 1)[1])
    L = simulate_sino(clean, np.zeros_like(clean), cat, seed)      # 零金属
    mu = iradon(L / PIXEL_MM, theta=THETA, filter_name=BASELINE_FILTER, circle=True)
    hu = np.clip(mu_to_hu(mu), LO, HI).astype(np.float32)
    hu[clean <= -500.0] = -1000.0
    np.save(cpath, hu)
    return hu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--outdir", default="/root/autodl-tmp/PRISM/v2.2/info_loss")
    ap.add_argument("--cache", default="/root/autodl-tmp/PRISM/v2.2/info_loss/_before_cache")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    meta_dir = os.path.join(args.data_dir, "meta")
    meta, by_size = {}, {s: [] for s in SIZES}
    for f in glob.glob(os.path.join(meta_dir, "*.json")):
        m = json.load(open(f))
        meta[m["sample_id"]] = m
        if m.get("category") in by_size:
            by_size[m["category"]].append(m["sample_id"])

    rows, curves = [], {}
    grid = np.arange(0, 401, 5)
    for size in SIZES:
        if not by_size[size]:
            continue
        best, best_m = None, -1.0
        for sid in by_size[size]:
            p = find_split(args.data_dir, sid, "Mask")
            if not p:
                continue
            mk = read_raw(p)
            s = float(mk[mk > MASK_EPS].sum())
            if s > best_m:
                best_m, best = s, sid
        if best is None:
            continue

        gt = read_raw(find_split(args.data_dir, best, "Target"))
        bl = read_raw(find_split(args.data_dir, best, "Baseline"))
        mask = read_raw(find_split(args.data_dir, best, "Mask"))
        tissue = (mask <= MASK_EPS) & (gt > AIR_HU)
        cf = clean_fbp(gt, meta[best].get('category','medium'), best, args.cache)

        e_before = np.abs(cf - gt)[tissue]
        e_after = np.abs(bl - gt)[tissue]
        m_b, m_a = float(e_before.mean()), float(e_after.mean())
        f50b, f50a = float((e_before > 50).mean()), float((e_after > 50).mean())
        f100b, f100a = float((e_before > 100).mean()), float((e_after > 100).mean())
        rows.append((size, best, int(tissue.sum()), best_m, m_b, m_a, f50b, f50a, f100b, f100a))
        curves[size] = (np.array([(e_before > t).mean() for t in grid]),
                        np.array([(e_after > t).mean() for t in grid]))
        print(f"[{size:6s}] {best} 金属={best_m:6.1f} 组织={int(tissue.sum()):6d} | "
              f"mean|Err| 前{m_b:6.1f}→后{m_a:6.1f} HU | "
              f">50HU 前{f50b:.3f}→后{f50a:.3f} | >100HU 前{f100b:.3f}→后{f100a:.3f}", flush=True)

    with open(os.path.join(args.outdir, "obs_info_loss.csv"), "w") as f:
        f.write("size,sid,n_tissue,metal,mean_err_before,mean_err_after,"
                "frac50_before,frac50_after,frac100_before,frac100_after\n")
        for r in rows:
            f.write(",".join([str(r[0]), r[1], str(r[2]), f"{r[3]:.2f}",
                              f"{r[4]:.2f}", f"{r[5]:.2f}", f"{r[6]:.4f}",
                              f"{r[7]:.4f}", f"{r[8]:.4f}", f"{r[9]:.4f}"]) + "\n")

    if rows:
        plt.rcParams.update({"font.family": "serif",
                             "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                             "font.size": 10, "axes.linewidth": 0.8})
        colors = {"small": "#8ecae6", "medium": "#219ebc", "large": "#fb8500", "giant": "#9d0208"}
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        for size in SIZES:
            if size not in curves:
                continue
            b, a = curves[size]
            ax.plot(grid, a, color=colors[size], lw=1.7, label=f"{size} (after)")
            ax.plot(grid, b, color=colors[size], lw=1.0, ls="--", alpha=0.7)
        ax.set_xlabel("observation error |Baseline − GT| threshold (HU)")
        ax.set_ylabel("fraction of tissue above threshold")
        ax.grid(color="#d9d9d9", lw=0.5, alpha=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=8, title="solid = after implant\n dashed = before",
                  title_fontsize=8)
        ax.set_title("observation-side information loss (non-metal tissue)", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "obs_info_loss.png"), dpi=150)
        plt.close(fig)
    print("[done]")


if __name__ == "__main__":
    main()
