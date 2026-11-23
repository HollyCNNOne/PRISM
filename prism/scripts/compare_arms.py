"""compare_arms.py — 机制实验多轴对比（不只看 MAE）。

对**同一批 held-out 序列、同一评分域**，比较多个 checkpoint 的：
    1. MAE / RMSE / PSNR / SSIM（图像质量）
    2. p50 / p90 / p99 / >400HU（误差分布与尾部）
    3. 金属窗口的 z 频带占比（低/中/高）
    4. 片间跳变 E_z（抖振）
    5. 面内高频（锐度）
用法：
    python compare_arms.py --ckpts a.pt,b.pt --names baseline,exp007 --max-seq 6
"""
import argparse, glob, json, os, sys
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, "/root/autodl-tmp/prism_framework")
from prism.data.dataset import make_theta_sequence_loader
from prism.models.m_network import MNetwork
from prism.models.find_net import FINDNet, FINDNetArgs
from prism.recursion.algorithm1 import forward_backward
from prism.training.trainer import SSMTrainer
from prism.utils.config import SSMConfig
from prism.eval.domain import tissue_mask
from prism.scripts.freq_recorder import z_band_proportions, select_pillars

D = "/root/autodl-tmp/PRISM/training_data_astra"


def read_raw(p):
    return np.fromfile(p, np.float32).reshape(512, 512)


def find_split(sid, sub):
    for sp in ("train", "val"):
        p = os.path.join(D, sp, sub, f"{sid}.raw")
        if os.path.exists(p):
            return p
    return None


def ssim_np(x, y, dr=3000.0, sigma=1.5):
    x = x.astype(np.float64); y = y.astype(np.float64)
    C1 = (0.01 * dr) ** 2; C2 = (0.03 * dr) ** 2
    mx = gaussian_filter(x, sigma); my = gaussian_filter(y, sigma)
    sxx = gaussian_filter(x * x, sigma) - mx ** 2
    syy = gaussian_filter(y * y, sigma) - my ** 2
    sxy = gaussian_filter(x * y, sigma) - mx * my
    s = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))
    return float(s.mean())


def build(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = ck["config"]
    m = MNetwork(nb=cfg.get("nb", 1), channels=cfg.get("channels", 32),
                 use_velocity=cfg.get("use_velocity", True),
                 shadow_r0=cfg.get("shadow_r0", 1.0 / 9.0),
                 use_z_curvature=cfg.get("use_z_curvature", False)).to(dev)
    fn = FINDNet(FINDNetArgs()).to(dev)
    m.load_state_dict(ck["model"]); fn.load_state_dict(ck["find_net"])
    config = SSMConfig(nb=cfg.get("nb", 1), channels=cfg.get("channels", 32),
                       d_ref=cfg["d_ref"], gain_mode=cfg.get("gain_mode", "kalman"),
                       findnet_lict_source=cfg.get("findnet_lict_source", "mu"),
                       backfill_r=cfg.get("backfill_r", 8),
                       recursion=cfg.get("recursion", "bidirectional"),
                       shadow_r0=cfg.get("shadow_r0", 1.0 / 9.0),
                       zfreq_gain=cfg.get("zfreq_gain", 0.0))
    return m, fn, config, SSMTrainer(m, config, device=dev, find_net=fn)


def run_seq(m, fn, config, trainer, sids, dev):
    loader = make_theta_sequence_loader(
        os.path.join(D, "train", "Target"), os.path.join(D, "train", "Baseline"),
        os.path.join(D, "train", "LI"), os.path.join(D, "train", "Mask"),
        os.path.join(D, "train", "Phys"))
    sm = loader(sids)
    Z = len(sids)
    th = [{k: v.unsqueeze(0).to(dev) for k, v in t.items()} for t in sm["thetas"]]
    mm = sm["metal_mask"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(dev)
    xh = sm["x_hat"].unsqueeze(0).to(dev)
    li = sm["li"].unsqueeze(0).to(dev)
    dv = xh.permute(0, 2, 1, 3, 4); lv = li.permute(0, 2, 1, 3, 4)
    dirty = [dv[:, :, z] for z in range(Z)]
    li_l = [lv[:, :, z] for z in range(Z)]
    masks = [mm[:, :, z] for z in range(Z)]
    obs = trainer._observer(dirty, masks, li_l)
    m.eval(); fn.eval()
    with torch.no_grad():
        res = forward_backward(m, th, config.d_ref, recursion=config.recursion,
                               observer=obs, gain_mode=config.gain_mode)
        out = trainer._final_output(res, dirty, masks, li_l)
    m.train(); fn.train()
    return out[0, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="逗号分隔")
    ap.add_argument("--names", default=None)
    ap.add_argument("--patient", default="patients_head2")
    ap.add_argument("--max-seq", type=int, default=6)
    ap.add_argument("--margin", type=int, default=8)
    ap.add_argument("--outdir", default="/root/autodl-tmp/PRISM/v2.2/arm_compare")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    ckpts = [c.strip() for c in args.ckpts.split(",") if c.strip()]
    names = (args.names.split(",") if args.names else
             [os.path.basename(c).replace(".pt", "") for c in ckpts])
    os.makedirs(args.outdir, exist_ok=True)
    dev = args.device

    meta = {}
    for f in glob.glob(os.path.join(D, "meta", "*.json")):
        m = json.load(open(f)); meta[m["sample_id"]] = m
    by = {}
    for sid, m in meta.items():
        if m.get("patient_id") == args.patient:
            by.setdefault(m["sequence_id"], []).append(sid)
    specs = [sorted(v, key=lambda x: int(x.rsplit("_", 1)[1])) for v in by.values()]
    specs = [s for s in specs if len(s) >= 20]
    specs.sort(key=lambda s: -len(s))
    specs = specs[:args.max_seq]
    print(f"[data] {len(specs)} 条 held-out 长序列", flush=True)

    rows = []
    for ck, nm in zip(ckpts, names):
        m, fn, config, trainer = build(ck, dev)
        agg = dict(mae=[], rmse=[], psnr=[], ssim=[], p50=[], p90=[], p99=[],
                   gt400=[], high=[], ez=[], hf=[], droot=[])
        for sid_list in specs:
            out = run_seq(m, fn, config, trainer, sid_list, dev)
            gt = np.stack([read_raw(find_split(s, "Target")) for s in sid_list])
            mk = np.stack([read_raw(find_split(s, "Mask")) for s in sid_list])
            tm = tissue_mask(mk, gt, r=8)
            e = np.abs(out - gt)[tm]
            agg["mae"].append(e.mean())
            agg["rmse"].append(np.sqrt((e ** 2).mean()))
            agg["psnr"].append(20 * np.log10(3000.0 / max(np.sqrt((e ** 2).mean()), 1e-6)))
            agg["ssim"].append(np.mean([ssim_np(gt[z], out[z]) for z in range(len(sid_list))]))
            agg["p50"].append(np.percentile(e, 50)); agg["p90"].append(np.percentile(e, 90))
            agg["p99"].append(np.percentile(e, 99)); agg["gt400"].append((e > 400).mean())
            # 金属窗口（金属片 ±margin）
            mz = np.where((mk > 1e-3).reshape(len(sid_list), -1).any(axis=1))[0]
            lo = max(0, mz.min() - args.margin); hi = min(len(sid_list), mz.max() + 1 + args.margin)
            ii = np.arange(lo, hi)
            sec = z_band_proportions  # noqa
            g_ = gt[ii]; o_ = out[ii]; t_ = tm[ii]
            pil = select_pillars(g_, mk[ii], n=4)
            gb = np.mean([z_band_proportions(g_[:, y, x]) for y, x in pil], axis=0)
            ob = np.mean([z_band_proportions(o_[:, y, x]) for y, x in pil], axis=0)
            agg["high"].append(ob[2] - gb[2])
            d = (o_[1:] - o_[:-1]); gg = (g_[1:] - g_[:-1]); sel = t_[1:]
            agg["ez"].append((d[sel] ** 2).mean() / max((gg[sel] ** 2).mean(), 1e-9))
            hf_g = hf_o = 0.0
            for zz in range(len(ii)):
                hg = g_[zz] - gaussian_filter(g_[zz], 3.0)
                ho = o_[zz] - gaussian_filter(o_[zz], 3.0)
                s = t_[zz]
                hf_g += float((hg[s] ** 2).sum()); hf_o += float((ho[s] ** 2).sum())
            agg["hf"].append(hf_o / max(hf_g, 1e-9))
        r = dict(name=nm, **{k: float(np.mean(v)) for k, v in agg.items()})
        rows.append(r)
        print(f"[{nm}] MAE {r['mae']:.1f} PSNR {r['psnr']:.1f} SSIM {r['ssim']:.3f} "
              f"p99 {r['p99']:.0f} >400 {r['gt400']*100:.2f}% | "
              f"Δhigh {r['high']:+.3f} Ez {r['ez']:.2f} HF {r['hf']:.2f}", flush=True)

    with open(os.path.join(args.outdir, "arm_compare.csv"), "w") as f:
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join([r["name"]] + [f"{r[k]:.4f}" for k in keys if k != "name"]) + "\n")
    print("[save]", os.path.join(args.outdir, "arm_compare.csv"))


if __name__ == "__main__":
    main()
