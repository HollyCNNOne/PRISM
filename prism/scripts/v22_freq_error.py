"""v22_freq_error.py — 一次性三路频域分析（GT / FIND-Net纯2D / FIND-Net+SSM）。

对 complete/large/medium 各一个 held-out 病人（完整头颅序列），选 3~5 根非金属柱子，
沿 z 做 1D FFT → 归一化占比 → 对数三等分（低/中/高）。画双面板：
上=z 信号（GT 填充带 + 纯2D + SSM 平滑曲线）；下=频率占比（填充面积 + 低中高带）。
输出 v2.2/fre_image/fre_error/{complete,large,medium}/。
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
from prism.scripts.freq_recorder import z_band_proportions, select_pillars

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BANDS = ["low", "mid", "high"]
COLORS = {"low": "#1f77b4", "mid": "#d62728", "high": "#f5c542"}


def load_seq(data_dir, patient, category):
    meta_dir = os.path.join(data_dir, "meta")
    by_seq = {}
    for f in glob.glob(os.path.join(meta_dir, "*.json")):
        m = json.load(open(f))
        if m.get("patient_id") == patient and m.get("case_category") == category:
            by_seq.setdefault(m["sequence_id"], []).append(m["sample_id"])
    if not by_seq:
        raise RuntimeError(f"{patient}/{category} 无 meta")
    best = max(by_seq, key=lambda s: len(by_seq[s]))
    return sorted(by_seq[best], key=lambda s: int(s.rsplit("_", 1)[1]))


def read_vol(data_dir, sids):
    H = W = 512
    Z = len(sids)
    vols = {}
    for name in ["Target", "Baseline", "LI", "Mask"]:
        v = np.zeros((Z, H, W), np.float32)
        for i, sid in enumerate(sids):
            p = None
            for sp in ("train", "val"):
                q = os.path.join(data_dir, sp, name, f"{sid}.raw")
                if os.path.exists(q):
                    p = q
                    break
            if p:
                v[i] = np.fromfile(p, np.float32).reshape(H, W)
        vols[name] = v
    return vols


def run_pure2d(find_net, dirty, li, metal, device):
    find_net.eval()
    Z = len(dirty)
    outs = []
    with torch.no_grad():
        for z in range(Z):
            ct = torch.from_numpy(dirty[z][None, None]).to(device)
            lc = torch.from_numpy(li[z][None, None]).to(device)
            mk = torch.from_numpy(metal[z][None, None]).to(device)
            outs.append(find_net.reconstruct(ct, lc, mk)[0, 0].cpu().numpy())
    return np.stack(outs)


def run_ssm(m_net, find_net, config, trainer, data_dir, sids, device):
    loader = make_theta_sequence_loader(
        os.path.join(data_dir, "train", "Target"),
        os.path.join(data_dir, "train", "Baseline"),
        os.path.join(data_dir, "train", "LI"),
        os.path.join(data_dir, "train", "Mask"),
        os.path.join(data_dir, "train", "Phys"))
    sample = loader(sids)
    Z = len(sids)
    thetas = [{k: v.unsqueeze(0).to(device) for k, v in t.items()} for t in sample["thetas"]]
    metal_mask = sample["metal_mask"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
    x_hat = sample["x_hat"].unsqueeze(0).to(device)
    li_t = sample["li"].unsqueeze(0).to(device)
    dv = x_hat.permute(0, 2, 1, 3, 4)
    dirty_l = [dv[:, :, z] for z in range(Z)]
    lv = li_t.permute(0, 2, 1, 3, 4)
    li_l = [lv[:, :, z] for z in range(Z)]
    masks = [metal_mask[:, :, z] for z in range(Z)]
    observer = trainer._observer(dirty_l, masks, li_l)
    m_net.eval()
    find_net.eval()
    with torch.no_grad():
        res = forward_backward(m_net, thetas, config.d_ref, recursion=config.recursion,
                               observer=observer, gain_mode=config.gain_mode)
        xh = trainer._final_output(res, dirty_l, masks, li_l)
    return xh[0, 0].cpu().numpy()


def plot_dual(gt, p2d, ssm, stats, cat, outdir):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "font.size": 10, "axes.linewidth": 0.8,
    })
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 6.6))
    z = np.arange(len(gt))
    ax1.fill_between(z, gt, color="#bbbbbb", alpha=0.4, label="GT", lw=0)
    ax1.plot(z, p2d, color="#1f77b4", lw=1.3, label="FIND-Net (2D)")
    ax1.plot(z, ssm, color="#d62728", lw=1.3, label="FIND-Net + SSM")
    ax1.set_ylabel("HU")
    ax1.grid(axis="y", color="#d9d9d9", lw=0.5, alpha=0.6)
    ax1.set_axisbelow(True)
    ax1.legend(frameon=False, fontsize=9, ncol=3)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    xs = np.arange(3)
    for name, c in [("GT", "#bbbbbb"), ("FINDNet", "#1f77b4"), ("SSM", "#d62728")]:
        ax2.fill_between(xs, 0, stats[name], alpha=0.28 if name != "GT" else 0.35,
                         color=c, label=name)
        ax2.plot(xs, stats[name], color=c, lw=1.4, marker="o", ms=4)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(["Low", "Mid", "High"])
    ax2.set_ylabel("z-frequency proportion")
    ax2.grid(axis="y", color="#d9d9d9", lw=0.5, alpha=0.6)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, fontsize=9, ncol=3)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax1.set_title(f"{cat} — z-axis frequency retention (one pillar)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(outdir, cat, "freq_retention.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--patient", default="patients_head2")
    ap.add_argument("--stage1", default="/root/autodl-tmp/PRISM/training_data_astra/findnet_stage1.pt")
    ap.add_argument("--stage2", default="/root/autodl-tmp/PRISM/training_data_astra/ssm_stage2.pt")
    ap.add_argument("--outdir", default="/root/autodl-tmp/PRISM/v2.2/fre_image/fre_error")
    ap.add_argument("--n-pillars", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device
    ck = torch.load(args.stage2, map_location=device, weights_only=False)
    cfg = ck["config"]
    m_net = MNetwork(nb=cfg.get("nb", 1), channels=cfg.get("channels", 32),
                     use_velocity=cfg.get("use_velocity", True)).to(device)
    find_net = FINDNet(FINDNetArgs()).to(device)
    m_net.load_state_dict(ck["model"])
    find_net.load_state_dict(ck["find_net"])
    config = SSMConfig(nb=cfg.get("nb", 1), channels=cfg.get("channels", 32),
                       d_ref=cfg["d_ref"], gain_mode=cfg.get("gain_mode", "kalman"),
                       findnet_lict_source=cfg.get("findnet_lict_source", "mu"),
                       backfill_r=cfg.get("backfill_r", 8),
                       recursion=cfg.get("recursion", "bidirectional"))
    trainer = SSMTrainer(m_net, config, device=device, find_net=find_net)

    ck1 = torch.load(args.stage1, map_location=device, weights_only=False)
    fn1 = FINDNet(FINDNetArgs()).to(device)
    fn1.load_state_dict(ck1["find_net"])

    for cat in ["complete", "large_residue", "medium_residue"]:
        try:
            sids = load_seq(args.data_dir, args.patient, cat)
        except RuntimeError as e:
            print(f"[skip] {e}")
            continue
        vols = read_vol(args.data_dir, sids)
        clean = vols["Target"]
        dirty = vols["Baseline"]
        li = vols["LI"]
        metal = vols["Mask"]
        pillars = select_pillars(clean, metal, n=args.n_pillars)
        gt = clean
        p2d = run_pure2d(fn1, dirty, li, metal, device)
        ssm = run_ssm(m_net, find_net, config, trainer, args.data_dir, sids, device)

        stats = {}
        for name, vol in [("GT", gt), ("FINDNet", p2d), ("SSM", ssm)]:
            props = np.array([z_band_proportions(vol[:, y, x]) for y, x in pillars])
            stats[name] = props.mean(axis=0)
        py, px = pillars[0]
        plot_dual(gt[:, py, px], p2d[:, py, px], ssm[:, py, px], stats, cat, args.outdir)
        os.makedirs(os.path.join(args.outdir, cat), exist_ok=True)
        with open(os.path.join(args.outdir, cat, "proportions.csv"), "w") as f:
            f.write("model,low,mid,high\n")
            for name in ["GT", "FINDNet", "SSM"]:
                f.write(f"{name},{stats[name][0]:.6f},{stats[name][1]:.6f},{stats[name][2]:.6f}\n")
        print(f"[done] {cat}: GT={stats['GT'].round(4)} FINDNet={stats['FINDNet'].round(4)} SSM={stats['SSM'].round(4)}")
    print("[all done]")


if __name__ == "__main__":
    main()
