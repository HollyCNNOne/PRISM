"""freq_stratified.py — 分层 z 频域分析（避免非金属片稀释）。

对 held-out 每个含金属序列，把 z 轴分成三段分别统计频带占比：
    metal_window : 金属切片 ± MARGIN 片（真正考验 SSM 的过渡区）
    clean_region : 金属窗口之外（应≈恒等）
    full         : 整条序列（旧口径，会被稀释）

比较 GT 与管线输出（FIND-Net + SSM）在三段的 低/中/高 频带占比与误差。
"""
import argparse, glob, json, os, sys
import numpy as np
import torch

_FR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _FR)

from prism.data.dataset import make_theta_sequence_loader
from prism.models.m_network import MNetwork
from prism.models.find_net import FINDNet, FINDNetArgs
from prism.recursion.algorithm1 import forward_backward
from prism.training.trainer import SSMTrainer
from prism.utils.config import SSMConfig
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from prism.eval.domain import tissue_mask
from prism.scripts.freq_recorder import z_band_proportions, select_pillars
from scipy.ndimage import gaussian_filter


def z_step_energy(vol, tissue=None):
    """片间跳变能量 E_z = mean( (x_z − x_{z−1})² )，只统计非金属体素。

    OUT/GT > 1 表示输出比真值更「跳」→ 片间抖振（伪造 z 突变），比模糊更坏。
    """
    if vol.shape[0] < 2:
        return float("nan")
    d = vol[1:] - vol[:-1]                      # [Z-1,H,W]
    if tissue is not None:
        return float((d[tissue[1:]] ** 2).mean())
    return float((d ** 2).mean())


def inplane_hf_energy(vol_slices, tissue=None, sigma=3.0):
    """2D 面内高频能量：mean( (x - Gaussian_σ(x))² )，只看非金属体素（mask<=1e-3）。"""
    tot, cnt = 0.0, 0
    for z, img in enumerate(vol_slices):
        hp = img - gaussian_filter(img, sigma)
        if tissue is not None:
            sel = tissue[z]                # 3D 布尔评分域（已排除金属+回填）
            tot += float((hp[sel] ** 2).sum()); cnt += int(sel.sum())
        else:
            tot += float((hp ** 2).sum()); cnt += hp.size
    return tot / max(cnt, 1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BANDS = ["low", "mid", "high"]


def read_raw(p):
    return np.fromfile(p, np.float32).reshape(512, 512)


def find_split(d, sid, sub):
    for sp in ("train", "val"):
        p = os.path.join(d, sp, sub, f"{sid}.raw")
        if os.path.exists(p):
            return p
    return None


def run_pipeline(m_net, find_net, config, trainer, data_dir, sids, device):
    loader = make_theta_sequence_loader(
        os.path.join(data_dir, "train", "Target"),
        os.path.join(data_dir, "train", "Baseline"),
        os.path.join(data_dir, "train", "LI"),
        os.path.join(data_dir, "train", "Mask"),
        os.path.join(data_dir, "train", "Phys"))
    sample = loader(sids)
    Z = len(sids)
    thetas = [{k: v.unsqueeze(0).to(device) for k, v in t.items()} for t in sample["thetas"]]
    mm = sample["metal_mask"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
    xh = sample["x_hat"].unsqueeze(0).to(device)
    li = sample["li"].unsqueeze(0).to(device)
    dv = xh.permute(0, 2, 1, 3, 4)
    dirty = [dv[:, :, z] for z in range(Z)]
    lv = li.permute(0, 2, 1, 3, 4)
    li_l = [lv[:, :, z] for z in range(Z)]
    masks = [mm[:, :, z] for z in range(Z)]
    obs = trainer._observer(dirty, masks, li_l)
    m_net.eval(); find_net.eval()
    with torch.no_grad():
        res = forward_backward(m_net, thetas, config.d_ref, recursion=config.recursion,
                               observer=obs, gain_mode=config.gain_mode)
        out = trainer._final_output(res, dirty, masks, li_l)
    m_net.train(); find_net.train()
    return out[0, 0].cpu().numpy(), mm[0, 0].cpu().numpy(), sample["X_clean"].numpy()[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--ckpt", default="/root/autodl-tmp/PRISM/training_data_astra/ssm_stage2_v22.pt")
    ap.add_argument("--patient", default="patients_head2")
    ap.add_argument("--margin", type=int, default=8)
    ap.add_argument("--max-seq", type=int, default=8)
    ap.add_argument("--max-z", type=int, default=0)
    ap.add_argument("--outdir", default="/root/autodl-tmp/PRISM/v2.2/fre_image/fre_strat")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    dev = args.device

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = ck["config"]
    m_net = MNetwork(nb=cfg.get("nb", 1), channels=cfg.get("channels", 32),
                     use_velocity=cfg.get("use_velocity", True),
                     shadow_r0=cfg.get("shadow_r0", 1.0 / 9.0),
                     use_z_curvature=cfg.get("use_z_curvature", False)).to(dev)
    find_net = FINDNet(FINDNetArgs()).to(dev)
    m_net.load_state_dict(ck["model"]); find_net.load_state_dict(ck["find_net"])
    config = SSMConfig(nb=cfg.get("nb", 1), channels=cfg.get("channels", 32),
                       d_ref=cfg["d_ref"], gain_mode=cfg.get("gain_mode", "kalman"),
                       findnet_lict_source=cfg.get("findnet_lict_source", "mu"),
                       backfill_r=cfg.get("backfill_r", 8),
                       recursion=cfg.get("recursion", "bidirectional"),
                       shadow_r0=cfg.get("shadow_r0", 1.0 / 9.0),
                       zfreq_gain=cfg.get("zfreq_gain", 0.0))
    trainer = SSMTrainer(m_net, config, device=dev, find_net=find_net)

    meta = {json.load(open(f))["sample_id"]: json.load(open(f))
            for f in glob.glob(os.path.join(args.data_dir, "meta", "*.json"))}
    by_seq = {}
    for sid, m in meta.items():
        if m.get("patient_id") == args.patient:
            by_seq.setdefault(m["sequence_id"], []).append(sid)
    specs = [sorted(v, key=lambda x: int(x.rsplit("_", 1)[1])) for v in by_seq.values()]
    specs = [s for s in specs if len(s) >= 20]
    specs.sort(key=lambda s: -len(s))
    specs = specs[:args.max_seq]
    print(f"[data] {len(specs)} 条 held-out 长序列", flush=True)

    agg = {"metal_window": [], "clean_region": [], "full": []}
    hf = {"metal_window": [], "clean_region": [], "full": []}
    ze = {"metal_window": [], "clean_region": [], "full": []}
    for spec in specs:
        if args.max_z > 0:
            spec = spec[:args.max_z]
        Z = len(spec)
        mk = np.stack([read_raw(find_split(args.data_dir, s, "Mask")) for s in spec])
        gt = np.stack([read_raw(find_split(args.data_dir, s, "Target")) for s in spec])
        metal_z = np.where((mk > 1e-3).reshape(Z, -1).any(axis=1))[0]
        if len(metal_z) == 0:
            continue
        lo = max(0, metal_z.min() - args.margin)
        hi = min(Z, metal_z.max() + 1 + args.margin)
        idx = {"metal_window": np.arange(lo, hi),
               "clean_region": np.setdiff1d(np.arange(Z), np.arange(lo, hi)),
               "full": np.arange(Z)}

        out, _, _ = run_pipeline(m_net, find_net, config, trainer, args.data_dir, spec, dev)
        pillars = select_pillars(gt, mk, n=4)
        for name, ii in idx.items():
            if len(ii) < 3:
                continue
            g = np.array([z_band_proportions(gt[ii, y, x]) for y, x in pillars])
            o = np.array([z_band_proportions(out[ii, y, x]) for y, x in pillars])
            agg[name].append((g.mean(0), o.mean(0)))
            tm = tissue_mask(mk[ii], gt[ii], r=8)      # 唯一评分域
            hf[name].append((inplane_hf_energy(gt[ii], tm), inplane_hf_energy(out[ii], tm)))
            ze[name].append((z_step_energy(gt[ii], tm), z_step_energy(out[ii], tm)))
        print(f"  seq {spec[0]} Z={Z} 金属片={len(metal_z)} 窗口=[{lo},{hi})", flush=True)

    print("\n=== 分层频带占比（跨序列平均）===")
    print(f"{'region':13s} {'band':5s} {'GT':>7s} {'OUT':>7s} {'|err|':>7s}")
    rows = []
    for name in ("metal_window", "clean_region", "full"):
        if not agg[name]:
            continue
        G = np.mean([a for a, b in agg[name]], axis=0)
        O = np.mean([b for a, b in agg[name]], axis=0)
        for i, b in enumerate(BANDS):
            print(f"{name:13s} {b:5s} {G[i]:7.3f} {O[i]:7.3f} {abs(G[i]-O[i]):7.3f}")
            rows.append((name, b, G[i], O[i]))
        print()

    print("=== 同一区域的 2D 面内高频能量（GT vs OUT）===")
    for name in ("metal_window", "clean_region", "full"):
        if not hf[name]:
            continue
        G = float(np.mean([a for a, b in hf[name]]))
        O = float(np.mean([b for a, b in hf[name]]))
        print(f"  {name:13s} GT={G:10.1f}  OUT={O:10.1f}  OUT/GT={O/max(G,1e-9):.3f}")
    print()

    print("=== 片间跳变能量 E_z = mean((x_z-x_{z-1})^2)  GT vs OUT ===")
    for name in ("metal_window", "clean_region", "full"):
        if not ze[name]:
            continue
        G = float(np.nanmean([a for a, b in ze[name]]))
        O = float(np.nanmean([b for a, b in ze[name]]))
        print(f"  {name:13s} GT={G:9.1f}  OUT={O:9.1f}  OUT/GT={O/max(G,1e-9):.3f}"
              f"  {'⚠ 抖振' if O > G*1.15 else ''}")
    print()

    with open(os.path.join(args.outdir, "freq_strat.csv"), "w") as f:
        f.write("region,band,gt,out\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6f}\n")

    # 图：三区域 × 三频带
    if rows:
        plt.rcParams.update({"font.family": "serif",
                             "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                             "font.size": 10})
        regions = [r for r in ("metal_window", "clean_region", "full") if agg[r]]
        fig, axes = plt.subplots(1, len(regions), figsize=(4.0 * len(regions), 3.4), sharey=True)
        if len(regions) == 1:
            axes = [axes]
        for ax, name in zip(axes, regions):
            idx = [i for i, r in enumerate(rows) if r[0] == name]
            g = [rows[i][2] for i in idx]; o = [rows[i][3] for i in idx]
            xs = np.arange(3)
            ax.plot(xs, g, "o-", color="#333333", lw=1.6, ms=5, label="GT")
            ax.plot(xs, o, "s--", color="#d62728", lw=1.6, ms=5, label="OUT")
            ax.set_xticks(xs); ax.set_xticklabels(["Low", "Mid", "High"])
            ax.set_title(name, fontsize=10)
            ax.grid(axis="y", color="#d9d9d9", lw=0.5, alpha=0.6); ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[0].set_ylabel("z-band proportion")
        axes[0].legend(frameon=False, fontsize=9)
        fig.suptitle("Stratified z-frequency: metal window vs clean region", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(os.path.join(args.outdir, "freq_strat.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("[save]", os.path.join(args.outdir, "freq_strat.png"))


if __name__ == "__main__":
    main()
