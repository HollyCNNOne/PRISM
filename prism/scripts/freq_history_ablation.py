"""freq_history_ablation.py — 历史截断因果实验（纯推理）。

同一序列、同一金属，只改「金属窗口之前保留多少干净片」：
    k ∈ {0, 5, 10, 20, 40}  →  子序列 = [z0-k, z1+1+TAIL)

评价窗口**固定**为 [z0, z1+1+TAIL)（金属片 + 之后 TAIL 片），与 k 无关 → 可比。

指标（评价窗口内，非金属体素）：
    d_high   高频占比差 (OUT−GT)          → <0 低通 / >0 高频上扬
    Ez_ratio 片间跳变能量 OUT/GT          → >1 抖振
    hf_ratio 2D 面内高频能量 OUT/GT       → <1 面内糊
    mae      组织 MAE (HU)
"""
import argparse, glob, json, os, sys
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KS = [0, 5, 10, 20, 40]
TAIL = 12
MASK_EPS = 1e-3


def read_raw(p):
    return np.fromfile(p, np.float32).reshape(512, 512)


def find_split(d, sid, sub):
    for sp in ("train", "val"):
        p = os.path.join(d, sp, sub, f"{sid}.raw")
        if os.path.exists(p):
            return p
    return None


def run(m_net, find_net, config, trainer, data_dir, sids, dev):
    loader = make_theta_sequence_loader(
        os.path.join(data_dir, "train", "Target"),
        os.path.join(data_dir, "train", "Baseline"),
        os.path.join(data_dir, "train", "LI"),
        os.path.join(data_dir, "train", "Mask"),
        os.path.join(data_dir, "train", "Phys"))
    sample = loader(sids)
    Z = len(sids)
    thetas = [{k: v.unsqueeze(0).to(dev) for k, v in t.items()} for t in sample["thetas"]]
    mm = sample["metal_mask"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(dev)
    xh = sample["x_hat"].unsqueeze(0).to(dev)
    li = sample["li"].unsqueeze(0).to(dev)
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
    return out[0, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--ckpt", default="/root/autodl-tmp/PRISM/training_data_astra/ssm_stage2_v22.pt")
    ap.add_argument("--patient", default="patients_head2")
    ap.add_argument("--max-seq", type=int, default=6)
    ap.add_argument("--outdir", default="/root/autodl-tmp/PRISM/v2.2/fre_image/fre_history")
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
                       shadow_r0=cfg.get("shadow_r0", 1.0 / 9.0))
    trainer = SSMTrainer(m_net, config, device=dev, find_net=find_net)

    meta = {}
    for f in glob.glob(os.path.join(args.data_dir, "meta", "*.json")):
        m = json.load(open(f)); meta[m["sample_id"]] = m
    by_seq = {}
    for sid, m in meta.items():
        if m.get("patient_id") == args.patient:
            by_seq.setdefault(m["sequence_id"], []).append(sid)
    specs = [sorted(v, key=lambda x: int(x.rsplit("_", 1)[1])) for v in by_seq.values()]
    specs = [s for s in specs if len(s) >= 40]
    specs.sort(key=lambda s: -len(s))
    specs = specs[:args.max_seq]
    print(f"[data] {len(specs)} 条序列", flush=True)

    rows = []
    for spec in specs:
        Z = len(spec)
        mk = np.stack([read_raw(find_split(args.data_dir, s, "Mask")) for s in spec])
        gt = np.stack([read_raw(find_split(args.data_dir, s, "Target")) for s in spec])
        metal_z = np.where((mk > MASK_EPS).reshape(Z, -1).any(axis=1))[0]
        if len(metal_z) == 0:
            continue
        z0, z1 = int(metal_z.min()), int(metal_z.max())
        ev_lo, ev_hi = z0, min(Z, z1 + 1 + TAIL)
        cat = meta[spec[0]].get("case_category", "?")
        for k in KS:
            st = max(0, z0 - k)
            sub = spec[st:ev_hi]
            if len(sub) < 6:
                continue
            out = run(m_net, find_net, config, trainer, args.data_dir, sub, dev)
            # 评价窗口在子序列中的位置
            e0 = z0 - st
            g = gt[ev_lo:ev_hi]; o = out[e0:e0 + (ev_hi - ev_lo)]
            m_ev = mk[ev_lo:ev_hi]
            tm_ev = tissue_mask(m_ev, g, r=8)          # 唯一评分域（排除金属+回填）
            pillars = select_pillars(g, m_ev, n=4)
            gb = np.mean([z_band_proportions(g[:, y, x]) for y, x in pillars], axis=0)
            ob = np.mean([z_band_proportions(o[:, y, x]) for y, x in pillars], axis=0)
            d = (o[1:] - o[:-1]); gg = (g[1:] - g[:-1])
            sel = tm_ev[1:]
            ez_r = float((d[sel] ** 2).mean() / max((gg[sel] ** 2).mean(), 1e-9))
            hf_g = hf_o = 0.0; n = 0
            for z in range(g.shape[0]):
                hg = g[z] - gaussian_filter(g[z], 3.0)
                ho = o[z] - gaussian_filter(o[z], 3.0)
                s = tm_ev[z]
                hf_g += float((hg[s] ** 2).sum()); hf_o += float((ho[s] ** 2).sum()); n += int(s.sum())
            hf_r = hf_o / max(hf_g, 1e-9)
            sel2 = tm_ev
            mae = float(np.abs(o - g)[sel2].mean())
            rows.append(dict(seq=spec[0], cat=cat, k=k, d_high=float(ob[2] - gb[2]),
                             gt_high=float(gb[2]), ez_ratio=ez_r, hf_ratio=hf_r, mae=mae))
            print(f"  {spec[0]} cat={cat} k={k:2d} 子序列={len(sub):3d} | "
                  f"d_high={ob[2]-gb[2]:+.3f} (GT {gb[2]:.3f}) Ez={ez_r:.2f} "
                  f"HF={hf_r:.2f} MAE={mae:.1f}", flush=True)

    import csv
    with open(os.path.join(args.outdir, "history_ablation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("[save] csv")

    # 图：k → 三个指标（按 case 上色，实线=均值）
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "font.size": 10})
    colors = {"complete": "#1f77b4", "large_residue": "#fb8500",
              "medium_residue": "#2ca02c", "small_residue": "#9d0208"}
    keys = [("d_high", "Δ high-band proportion (OUT−GT)"),
            ("ez_ratio", "E_z ratio (OUT/GT)  — >1 = flicker"),
            ("hf_ratio", "in-plane HF ratio (OUT/GT) — <1 = blurred")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (key, lab) in zip(axes, keys):
        for cat in sorted(set(r["cat"] for r in rows)):
            xs = sorted(set(r["k"] for r in rows if r["cat"] == cat))
            ys = [np.mean([r[key] for r in rows if r["cat"] == cat and r["k"] == k]) for k in xs]
            ax.plot(xs, ys, "o-", color=colors.get(cat, "#777"), lw=1.6, ms=5, label=cat)
        if key == "ez_ratio":
            ax.axhline(1.0, color="#999", ls=":", lw=1)
        if key == "hf_ratio":
            ax.axhline(1.0, color="#999", ls=":", lw=1)
        if key == "d_high":
            ax.axhline(0.0, color="#999", ls=":", lw=1)
        ax.set_xlabel("clean slices kept BEFORE the metal window (k)")
        ax.set_title(lab, fontsize=9)
        ax.grid(color="#d9d9d9", lw=0.5, alpha=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("History-truncation ablation (same anatomy, same metal, only history varies)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(args.outdir, "history_ablation.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[save] png")


if __name__ == "__main__":
    main()
