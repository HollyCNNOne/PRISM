"""v22_zdistance.py — 捕获率 vs "到最近无金属片的距离"（SSM-1 的可证伪推论）。

机制假设
    μ_z = (1−Δ)·μ_{z-1} + Δ·μ_cand
    携带项每片乘 (1−Δ)；若 Δ≈0.9，则每片 ×0.1。
    ⇒ 离"观测未被毁的片"越远，neighbor 携带的细节衰减越多。

推论
    在近环内，μ 与 out 的带限相关应随 d(z)（到最近"无金属片"的片距）**单调下降**。
    若 d 与相关**无关**，则上述对 Δ 传递的理解有误，需重看。

做法
    对每条 held-out 序列：
      · m(z) = 该片金属体素数；m(z) ≤ m_thr·max(m) 记为"信息片"
      · d(z) = min |z − z'|，z' 为信息片
      · 只统计"该片有 near 环带体素"的切片
      · 逐片算近环内的带限相关（μ / out / zprior vs GT），按 d 分箱
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
SIGMAS = (1.0, 4.0, 16.0)


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


def band_corr_slice(a, b, sel, sigma):
    if sel.sum() < 50:
        return np.nan
    ha = a - gaussian_filter(a, sigma)
    hb = b - gaussian_filter(b, sigma)
    x = ha[sel].astype(np.float64)
    y = hb[sel].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--ckpt", default="/root/autodl-tmp/PRISM/training_data_astra/ssm_stage2_v22.pt")
    ap.add_argument("--patient", default="patients_head2")
    ap.add_argument("--max-seq", type=int, default=6)
    ap.add_argument("--m-thr-frac", type=float, default=0.1,
                    help="m(z) ≤ frac·max(m) 视为「信息片」")
    ap.add_argument("--outdir", default="/root/autodl-tmp/PRISM/v2.2/zdistance")
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

    rows = []
    for spec in specs:
        Z = len(spec)
        gt = np.stack([read_raw(find_split(args.data_dir, s, "Target")) for s in spec])
        mk = np.stack([read_raw(find_split(args.data_dir, s, "Mask")) for s in spec])
        li = np.stack([read_raw(find_split(args.data_dir, s, "LI")) for s in spec])

        mz = (mk > MASK_EPS).reshape(Z, -1).sum(axis=1).astype(np.float64)
        if mz.max() <= 0:
            continue
        info = mz <= args.m_thr_frac * mz.max()
        idx_info = np.where(info)[0]
        if len(idx_info) == 0:
            continue

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
        for z in range(Z):
            if mz[z] <= MASK_EPS:
                continue
            core = mk[z] > MASK_EPS
            near = (_dil(core, NEAR_OUT) & (~_dil(core, NEAR_IN))) & tm[z]
            if near.sum() < 200:
                continue
            d = int(np.min(np.abs(idx_info - z)))
            rec = dict(seq=spec[0], z=z, d=d, metal=float(mz[z]))
            for name, vol in (("mu", mu_np), ("out", out_np)):
                for s in SIGMAS:
                    rec["%s_s%d" % (name, int(s))] = band_corr_slice(vol[z], gt[z], near, s)
            rows.append(rec)
        print("  seq %s Z=%d 金属片=%d 信息片=%d done"
              % (spec[0], Z, int((mz > MASK_EPS).sum()), len(idx_info)), flush=True)

    if not rows:
        print("[WARN] 无有效切片")
        return

    import csv
    outp = os.path.join(args.outdir, "zdistance.csv")
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("[save]", outp)

    def nm(v):
        v = [x for x in v if x == x]
        return float(np.mean(v)) if v else float("nan")

    print("\n=== 按到最近信息片的距离 d 分箱（近环内，逐片）===")
    hdr = "%-6s %5s " % ("d", "n")
    hdr += " ".join("%9s" % ("%s_s%d" % (nm_, s)) for nm_ in ("mu", "out") for s in SIGMAS)
    print(hdr)
    bins = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 10 ** 6)]
    for lo, hi in bins:
        sub = [r for r in rows if lo <= r["d"] <= hi]
        if not sub:
            continue
        line = "%-6s %5d " % (("%d" % lo) if lo == hi else ("%d-%d" % (lo, hi)), len(sub))
        for nm_ in ("mu", "out"):
            for s in SIGMAS:
                line += " %9.3f" % nm([r["%s_s%d" % (nm_, int(s))] for r in sub])
        print(line)


if __name__ == "__main__":
    main()
