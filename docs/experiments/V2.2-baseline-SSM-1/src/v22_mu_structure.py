"""v22_mu_structure.py — 先验 μ 到底带不带细结构？（决定"改 loss"还是"改架构"）

对 held-out 序列跑完整管线，把 **融合后的先验 μ** 取出来，与 GT 比带限相关：

    mu      : res.fused 里的 mu（SSM 交给 FIND-Net 的先验）
    out     : 最终输出 x̂_z
    obs     : Baseline（观测）
    zprior  : GT 上一片（z 上界）

区域：near / streak / far（与 -3 同定义）。

判据：
    mu 的 σ=1 相关 ≈ 0        → 信息在递归这一步就丢了 → 改架构
    mu 明显 > 0 但 out 低     → 信息在 FIND-Net 出口丢了 → 改 loss
    mu 接近 zprior 0.28       → 信息一直都在 → 改评估
"""
import argparse, glob, json, os, sys
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, maximum_filter

_FR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _FR)
from prism.data.dataset import make_theta_sequence_loader
from prism.models.m_network import MNetwork
from prism.models.find_net import FINDNet, FINDNetArgs
from prism.recursion.algorithm1 import forward_backward
from prism.training.trainer import SSMTrainer
from prism.utils.config import SSMConfig
from prism.eval.domain import tissue_mask

MASK_EPS = 1e-3
NEAR_IN, NEAR_OUT = 8, 30
STREAK_PCT = 90.0
SIGMAS = (1.0, 2.0, 4.0, 8.0, 16.0)


def read_raw(p):
    return np.fromfile(p, np.float32).reshape(512, 512)


def find_split(d, sid, sub):
    for sp in ("train", "val"):
        p = os.path.join(d, sp, sub, sid + ".raw")
        if os.path.exists(p):
            return p
    return None


def _dil(b, r):
    return maximum_filter(b, size=(2 * r + 1, 2 * r + 1), mode="constant", cval=False)


def build_regions(mk, li, tm):
    near = np.zeros_like(tm)
    for z in range(mk.shape[0]):
        core = mk[z] > MASK_EPS
        near[z] = (_dil(core, NEAR_OUT) & (~_dil(core, NEAR_IN))) & tm[z]
    a = np.abs(li)
    thr = np.percentile(a[tm], STREAK_PCT) if tm.any() else np.inf
    streak = (a > thr) & tm & (~near)
    far = tm & (~near) & (~streak)
    return {"near": near, "streak": streak, "far": far}


def band_corr(x, y, sel, sigma):
    vals = []
    for z in range(x.shape[0]):
        s = sel[z]
        if s.sum() < 50:
            continue
        hx = x[z] - gaussian_filter(x[z], sigma)
        hy = y[z] - gaussian_filter(y[z], sigma)
        a = hx[s].astype(np.float64)
        b = hy[s].astype(np.float64)
        a -= a.mean()
        b -= b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        if den > 0:
            vals.append(float((a * b).sum() / den))
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--ckpt", default="/root/autodl-tmp/PRISM/training_data_astra/ssm_stage2_v22.pt")
    ap.add_argument("--patient", default="patients_head2")
    ap.add_argument("--max-seq", type=int, default=6)
    ap.add_argument("--outdir", default="/root/autodl-tmp/PRISM/v2.2/mu_structure")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    dev = args.device

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = ck.get("config", {})
    m = MNetwork(nb=cfg.get("nb", 1), channels=cfg.get("channels", 32),
                 use_velocity=cfg.get("use_velocity", True),
                 shadow_r0=cfg.get("shadow_r0", 1.0 / 9.0),
                 use_z_curvature=cfg.get("use_z_curvature", False)).to(dev)
    fn = FINDNet(FINDNetArgs()).to(dev)
    m.load_state_dict(ck["model"])
    fn.load_state_dict(ck["find_net"])
    conf = SSMConfig(nb=cfg.get("nb", 1), channels=cfg.get("channels", 32),
                     d_ref=cfg["d_ref"], gain_mode=cfg.get("gain_mode", "kalman"),
                     findnet_lict_source=cfg.get("findnet_lict_source", "mu"),
                     backfill_r=cfg.get("backfill_r", 8),
                     recursion=cfg.get("recursion", "bidirectional"),
                     shadow_r0=cfg.get("shadow_r0", 1.0 / 9.0),
                     zfreq_gain=cfg.get("zfreq_gain", 0.0))
    tr = SSMTrainer(m, conf, device=dev, find_net=fn)

    meta = {}
    for f in glob.glob(os.path.join(args.data_dir, "meta", "*.json")):
        mm = json.load(open(f))
        meta[mm["sample_id"]] = mm
    by = {}
    for sid, mm in meta.items():
        if mm.get("patient_id") == args.patient:
            by.setdefault(mm["sequence_id"], []).append(sid)
    specs = [sorted(v, key=lambda x: int(x.rsplit("_", 1)[1])) for v in by.values()]
    specs = [s for s in specs if len(s) >= 20]
    specs.sort(key=lambda s: -len(s))
    specs = specs[:args.max_seq]
    print("[data] %d 条 held-out 长序列" % len(specs), flush=True)

    REG = ("near", "streak", "far")
    SRC = ("mu", "out", "obs", "zprior")
    AGG = {r: {k: {s: [] for s in SIGMAS} for k in SRC} for r in REG}

    for spec in specs:
        Z = len(spec)
        gt = np.stack([read_raw(find_split(args.data_dir, s, "Target")) for s in spec])
        mk = np.stack([read_raw(find_split(args.data_dir, s, "Mask")) for s in spec])
        li = np.stack([read_raw(find_split(args.data_dir, s, "LI")) for s in spec])
        bl = np.stack([read_raw(find_split(args.data_dir, s, "Baseline")) for s in spec])

        loader = make_theta_sequence_loader(
            os.path.join(args.data_dir, "train", "Target"),
            os.path.join(args.data_dir, "train", "Baseline"),
            os.path.join(args.data_dir, "train", "LI"),
            os.path.join(args.data_dir, "train", "Mask"),
            os.path.join(args.data_dir, "train", "Phys"))
        sm = loader(spec)
        th = [{k: v.unsqueeze(0).to(dev) for k, v in t.items()} for t in sm["thetas"]]
        mm_ = sm["metal_mask"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(dev)
        xh = sm["x_hat"].unsqueeze(0).to(dev)
        liv = sm["li"].unsqueeze(0).to(dev)
        dv = xh.permute(0, 2, 1, 3, 4)
        lv = liv.permute(0, 2, 1, 3, 4)
        dirty = [dv[:, :, z] for z in range(Z)]
        li_l = [lv[:, :, z] for z in range(Z)]
        masks = [mm_[:, :, z] for z in range(Z)]
        obs = tr._observer(dirty, masks, li_l)
        m.eval()
        fn.eval()
        with torch.no_grad():
            res = forward_backward(m, th, conf.d_ref, recursion=conf.recursion,
                                   observer=obs, gain_mode=conf.gain_mode)
            out = tr._final_output(res, dirty, masks, li_l)
            mu = tr._stack([st.mu for st in res.fused])
        m.train()
        fn.train()
        mu_np = mu[0, 0].cpu().numpy()
        out_np = out[0, 0].cpu().numpy()

        tm = tissue_mask(mk, gt, r=8)
        regions = build_regions(mk, li, tm)
        zp = np.concatenate([gt[:1], gt[:-1]], axis=0)
        cand = {"mu": mu_np, "out": out_np, "obs": bl, "zprior": zp}
        for rn, sel in regions.items():
            if sel.sum() < 2000:
                continue
            for k in SRC:
                for s in SIGMAS:
                    AGG[rn][k][s].append(band_corr(cand[k], gt, sel, s))
        print("  seq %s Z=%d done" % (spec[0], Z), flush=True)

    def nm(v):
        return float(np.nanmean(v)) if len(v) else float("nan")

    print("\n=== 带限相关 corr(HP_sigma(src), HP_sigma(GT)) ===")
    hdr = " ".join("%7s" % ("s=" + str(int(s))) for s in SIGMAS)
    print("%-8s %-8s %s" % ("region", "src", hdr))
    for r in REG:
        for k in SRC:
            vals = [nm(AGG[r][k][s]) for s in SIGMAS]
            print("%-8s %-8s %s" % (r, k, " ".join("%7.3f" % v for v in vals)))
        print()

    outp = os.path.join(args.outdir, "mu_structure.csv")
    with open(outp, "w") as f:
        f.write("region,src," + ",".join("sigma%d" % int(s) for s in SIGMAS) + "\n")
        for r in REG:
            for k in SRC:
                f.write("%s,%s," % (r, k) +
                        ",".join("%.4f" % nm(AGG[r][k][s]) for s in SIGMAS) + "\n")
    print("[save]", outp)


if __name__ == "__main__":
    main()
