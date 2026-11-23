"""v22_findnet_obs_error.py — 用 v2.1 FIND-Net（No-GF）处理四个典型切片，
按「观测端误差」口径记录：|处理结果 − GT| vs |Baseline − GT|（非金属组织）。

对照三行：
  1. cleanFBP   —— 植入前（零金属重建，只有重建模糊）
  2. Baseline   —— 植入后未处理（模糊 + 伪影）
  3. FIND-Net   —— 植入后用 v2.1 FIND-Net 去伪影
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from prism.eval.domain import tissue_mask
import argparse, glob, json, os, sys
import numpy as np
import torch

_FR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _FR)

from prism.models.find_net import FINDNet, FINDNetArgs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIZES = ["small", "medium", "large", "giant"]
MASK_EPS = 1e-3
AIR_HU = -500.0


def read_raw(p):
    return np.fromfile(p, np.float32).reshape(512, 512)


def find_split(d, sid, sub):
    for sp in ("train", "val"):
        p = os.path.join(d, sp, sub, f"{sid}.raw")
        if os.path.exists(p):
            return p
    return None


def _backfill(pred, baseline, mask, r=8):
    """与 trainer._backfill 完全一致：金属区(>0.5)膨胀 r px 后整块用 Baseline 覆盖。"""
    import torch.nn.functional as F
    b = torch.from_numpy((mask > 0.5).astype(np.float32))[None, None]
    dil = F.max_pool2d(b, kernel_size=2 * r + 1, stride=1, padding=r)[0, 0].numpy()
    return (1.0 - dil) * pred + dil * baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--ckpt", default="/root/autodl-tmp/PRISM/training_data_astra/findnet_stage1_v2.1.pt")
    ap.add_argument("--outdir", default="/root/autodl-tmp/PRISM/v2.2/info_loss")
    ap.add_argument("--cache", default="/root/autodl-tmp/PRISM/v2.2/info_loss/_before_cache")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    net = FINDNet(FINDNetArgs()).to(args.device)
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    net.load_state_dict(ck["find_net"])
    net.eval()
    print(f"[load] {args.ckpt} (epoch {ck.get('config', {}).get('epoch')})", flush=True)

    meta = {}
    for f in glob.glob(os.path.join(args.data_dir, "meta", "*.json")):
        m = json.load(open(f))
        meta[m["sample_id"]] = m

    # 每个 size 取金属量最大的切片（与 info-loss 实验同一批切片，可直接对照）
    picks = {"small": "sample_003090", "medium": "sample_003169",
             "large": "sample_003260", "giant": "sample_001308"}
    rows, curves = [], {}
    grid = np.arange(0, 401, 5)

    for size in SIZES:
        sid = picks[size]
        gt = read_raw(find_split(args.data_dir, sid, "Target"))
        bl = read_raw(find_split(args.data_dir, sid, "Baseline"))
        li = read_raw(find_split(args.data_dir, sid, "LI"))
        mk = read_raw(find_split(args.data_dir, sid, "Mask"))
        tissue = tissue_mask(mk, gt, r=8)

        cf = None
        p = os.path.join(args.cache, f"{sid}_cleanFBP.npy")
        if os.path.exists(p):
            cf = np.load(p).astype(np.float32)

        with torch.no_grad():
            t = lambda a: torch.from_numpy(a[None, None].astype(np.float32)).to(args.device)
            x = net.reconstruct(t(bl), t(li), t(mk))[0, 0].cpu().numpy()
        x = _backfill(x, bl, mk)

        e_bl = np.abs(bl - gt)[tissue]
        e_fn = np.abs(x - gt)[tissue]
        e_cf = np.abs(cf - gt)[tissue] if cf is not None else None

        def stat(e):
            return (float(e.mean()), float((e > 50).mean()), float((e > 100).mean()))

        m_bl, f50_bl, f100_bl = stat(e_bl)
        m_fn, f50_fn, f100_fn = stat(e_fn)
        m_cf = stat(e_cf) if e_cf is not None else (float("nan"),) * 3

        rows.append((size, sid, int(tissue.sum()), m_cf, (m_bl, f50_bl, f100_bl), (m_fn, f50_fn, f100_fn)))
        curves[size] = (np.array([(e_bl > v).mean() for v in grid]),
                        np.array([(e_fn > v).mean() for v in grid]))
        print(f"[{size:6s}] {sid} | cleanFBP {m_cf[0]:6.1f} | Baseline {m_bl:6.1f} HU "
              f"(>50:{f50_bl:.3f} >100:{f100_bl:.3f}) -> FIND-Net {m_fn:6.1f} HU "
              f"(>50:{f50_fn:.3f} >100:{f100_fn:.3f})", flush=True)

    with open(os.path.join(args.outdir, "findnet_obs_error.csv"), "w") as f:
        f.write("size,sid,n_tissue,cleanFBP_mean,base_mean,base_gt50,base_gt100,"
                "findnet_mean,findnet_gt50,findnet_gt100\n")
        for size, sid, n, cf, b, fn in rows:
            f.write(f"{size},{sid},{n},{cf[0]:.2f},{b[0]:.2f},{b[1]:.4f},{b[2]:.4f},"
                    f"{fn[0]:.2f},{fn[1]:.4f},{fn[2]:.4f}\n")

    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "font.size": 10, "axes.linewidth": 0.8})
    colors = {"small": "#8ecae6", "medium": "#219ebc", "large": "#fb8500", "giant": "#9d0208"}
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for size in SIZES:
        if size not in curves:
            continue
        b, fn = curves[size]
        ax.plot(grid, b, color=colors[size], lw=1.0, ls="--", alpha=0.75)
        ax.plot(grid, fn, color=colors[size], lw=1.8, label=f"{size}")
    ax.set_xlabel("observation error |x − GT| threshold (HU)")
    ax.set_ylabel("fraction of tissue above threshold")
    ax.grid(color="#d9d9d9", lw=0.5, alpha=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, title="solid = FIND-Net (v2.1)\ndashed = Baseline",
              title_fontsize=8)
    ax.set_title("observation error after FIND-Net (non-metal tissue)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "findnet_obs_error.png"), dpi=150)
    plt.close(fig)
    print("[done]")


if __name__ == "__main__":
    main()
