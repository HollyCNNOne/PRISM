"""v22_clinical_usability.py — 与训练目标正交的「临床可用性」度量（exp014）。

为什么需要它：
    MAE 就是训练目标（L_down）。用它证明「可用」是循环论证——
    模型被按 MAE 训过，当然能在 MAE 上给出合理数字。
    一个均值填充的图像，低频趋势对了，MAE 就好看，但结构信息为零。
    本脚本换四把与 MAE 正交的尺子。

四个度量（全部按区域 near / streak / far 报告）：
  ① 多尺度带限相关  corr(HP_σ(OUT), HP_σ(GT))，σ = 1,2,4,8,16
     → 输出到底带着哪一档频率的**真信息**（不关心能量大小，只关心对不对）
  ② edge preservation      在 |∇GT| > p90 的边缘上，mean|∇OUT|/mean|∇GT| + 方向余弦
     → 真边界还在不在（临床最关心）
  ③ SSIM 三因子     l（亮度）/ c（对比）/ s（结构），分区统计
     → 把「亮度对」和「结构对」分开
  ④ 误差空间自相关  误差图的相关长度（lag-1 与 lag-8）
     → 误差是「成片替换组织」还是「随机噪声」

区域定义只用推理期输入（金属 mask、LI），不用 GT。

用法：
    python v22_clinical_usability.py --data-dir ... --ckpt ... --outdir ... --device cuda
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MASK_EPS = 1e-3
NEAR_IN, NEAR_OUT = 8, 30
STREAK_PCT = 90.0
SIGMAS = (1.0, 2.0, 4.0, 8.0, 16.0)
EDGE_PCT = 90.0


def read_raw(p):
    return np.fromfile(p, np.float32).reshape(512, 512)


def find_split(d, sid, sub):
    for sp in ("train", "val"):
        p = os.path.join(d, sp, sub, f"{sid}.raw")
        if os.path.exists(p):
            return p
    return None


def dilate2d(binm, r):
    if r <= 0:
        return binm
    return maximum_filter(binm, size=(2 * r + 1, 2 * r + 1), mode="constant", cval=False)


def build_regions(mk, li, tm):
    near = np.zeros_like(tm)
    for z in range(mk.shape[0]):
        core = mk[z] > MASK_EPS
        ring = dilate2d(core, NEAR_OUT) & (~dilate2d(core, NEAR_IN))
        near[z] = ring & tm[z]
    a = np.abs(li)
    thr = np.percentile(a[tm], STREAK_PCT) if tm.any() else np.inf
    streak = (a > thr) & tm & (~near)
    far = tm & (~near) & (~streak)
    return {"near": near, "streak": streak, "far": far}


def corr_sel(a, b, sel):
    x = a[sel].astype(np.float64); y = b[sel].astype(np.float64)
    if x.size < 50:
        return np.nan
    x -= x.mean(); y -= y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else np.nan


def band_corr(out, gt, sel, sigma):
    """HP_σ 带限相关：先各自去掉 σ 尺度的低频，再在分区内求相关。"""
    vals = []
    for z in range(out.shape[0]):
        if sel[z].sum() < 50:
            continue
        ho = out[z] - gaussian_filter(out[z], sigma)
        hg = gt[z] - gaussian_filter(gt[z], sigma)
        c = corr_sel(ho, hg, sel[z])
        if not np.isnan(c):
            vals.append(c)
    return float(np.mean(vals)) if vals else np.nan


def edge_metrics(out, gt, sel):
    """在 GT 强边缘处：梯度幅值保持率 + 方向余弦。"""
    mag_r, cos_v = [], []
    for z in range(out.shape[0]):
        s = sel[z]
        if s.sum() < 50:
            continue
        gy_g, gx_g = np.gradient(gt[z])
        gy_o, gx_o = np.gradient(out[z])
        mg = np.hypot(gx_g, gy_g)
        mo = np.hypot(gx_o, gy_o)
        thr = np.percentile(mg[s], EDGE_PCT)
        e = s & (mg > thr)
        if e.sum() < 50:
            continue
        mag_r.append(float(mo[e].mean() / max(mg[e].mean(), 1e-9)))
        num = (gx_o[e] * gx_g[e] + gy_o[e] * gy_g[e]).sum()
        den = np.sqrt((gx_o[e] ** 2 + gy_o[e] ** 2).sum() * (gx_g[e] ** 2 + gy_g[e] ** 2).sum())
        cos_v.append(float(num / den) if den > 0 else np.nan)
    return (float(np.nanmean(mag_r)) if mag_r else np.nan,
            float(np.nanmean(cos_v)) if cos_v else np.nan)


def ssim_components(out, gt, sel, sigma=1.5, dr=3000.0):
    """SSIM = l·c·s 三因子，分区统计均值。"""
    C1 = (0.01 * dr) ** 2; C2 = (0.03 * dr) ** 2
    L, C, S = [], [], []
    for z in range(out.shape[0]):
        s = sel[z]
        if s.sum() < 50:
            continue
        x = out[z].astype(np.float64); y = gt[z].astype(np.float64)
        mx = gaussian_filter(x, sigma); my = gaussian_filter(y, sigma)
        sxx = gaussian_filter(x * x, sigma) - mx ** 2
        syy = gaussian_filter(y * y, sigma) - my ** 2
        sxy = gaussian_filter(x * y, sigma) - mx * my
        l = (2 * mx * my + C1) / (mx ** 2 + my ** 2 + C1)
        c = (2 * np.sqrt(np.maximum(sxx, 0) * np.maximum(syy, 0)) + C2) / (sxx + syy + C2)
        st = (sxy + C2 / 2) / (np.sqrt(np.maximum(sxx, 0) * np.maximum(syy, 0)) + C2 / 2)
        L.append(l[s].mean()); C.append(c[s].mean()); S.append(st[s].mean())
    return (float(np.mean(L)) if L else np.nan,
            float(np.mean(C)) if C else np.nan,
            float(np.mean(S)) if S else np.nan)


def err_autocorr(out, gt, sel, lag):
    """误差图的空间自相关（lag 像素；3D 上只在同片内沿 x/y 错位）。"""
    vals = []
    for z in range(out.shape[0]):
        s = sel[z]
        if s.sum() < 200:
            continue
        e = (out[z] - gt[z]) * s
        a = e[:, :-lag] if lag > 0 else e
        b = e[:, lag:] if lag > 0 else e
        m = s[:, :-lag] & s[:, lag:] if lag > 0 else s
        if m.sum() < 200:
            continue
        x = a[m].astype(np.float64); y = b[m].astype(np.float64)
        if x.std() < 1e-6 or y.std() < 1e-6:
            continue
        vals.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(vals)) if vals else np.nan


def run_pipeline(m_net, find_net, config, trainer, data_dir, sids, dev):
    loader = make_theta_sequence_loader(
        os.path.join(data_dir, "train", "Target"), os.path.join(data_dir, "train", "Baseline"),
        os.path.join(data_dir, "train", "LI"), os.path.join(data_dir, "train", "Mask"),
        os.path.join(data_dir, "train", "Phys"))
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
    m_net.eval(); find_net.eval()
    with torch.no_grad():
        res = forward_backward(m_net, th, config.d_ref, recursion=config.recursion,
                               observer=obs, gain_mode=config.gain_mode)
        out = trainer._final_output(res, dirty, masks, li_l)
        fn_2d = [obs(dirty[z], z) for z in range(Z)]
    m_net.train(); find_net.train()
    return (out[0, 0].cpu().numpy(),
            torch.stack(fn_2d, dim=2)[0, 0].cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--ckpt", default="/root/autodl-tmp/PRISM/training_data_astra/ssm_stage2_v22.pt")
    ap.add_argument("--patient", default="patients_head2")
    ap.add_argument("--max-seq", type=int, default=6)
    ap.add_argument("--outdir", default="/root/autodl-tmp/PRISM/v2.2/clinical")
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

    REG = ("near", "streak", "far")
    MODELS = ("findnet", "ssm")
    BAND = {m: {r: {s: [] for s in SIGMAS} for r in REG} for m in MODELS}
    EDGE = {m: {r: {"mag": [], "cos": []} for r in REG} for m in MODELS}
    SSIMC = {m: {r: {"l": [], "c": [], "s": []} for r in REG} for m in MODELS}
    AC = {m: {r: {lg: [] for lg in (1, 8)} for r in REG} for m in MODELS}

    for spec in specs:
        Z = len(spec)
        gt = np.stack([read_raw(find_split(args.data_dir, s, "Target")) for s in spec])
        mk = np.stack([read_raw(find_split(args.data_dir, s, "Mask")) for s in spec])
        li = np.stack([read_raw(find_split(args.data_dir, s, "LI")) for s in spec])
        out, fn = run_pipeline(m_net, find_net, config, trainer, args.data_dir, spec, dev)
        fn = fn[:Z]
        tm = tissue_mask(mk, gt, r=8)
        regions = build_regions(mk, li, tm)
        for mname, vol in (("findnet", fn), ("ssm", out)):
            for rname, sel in regions.items():
                if sel.sum() < 2000:
                    continue
                for s in SIGMAS:
                    BAND[mname][rname][s].append(band_corr(vol, gt, sel, s))
                mg, cs = edge_metrics(vol, gt, sel)
                EDGE[mname][rname]["mag"].append(mg); EDGE[mname][rname]["cos"].append(cs)
                l, c, st = ssim_components(vol, gt, sel)
                SSIMC[mname][rname]["l"].append(l)
                SSIMC[mname][rname]["c"].append(c)
                SSIMC[mname][rname]["s"].append(st)
                for lg in (1, 8):
                    AC[mname][rname][lg].append(err_autocorr(vol, gt, sel, lg))
        print(f"  seq {spec[0]} Z={Z} done", flush=True)

    nm = lambda x: float(np.nanmean(x)) if len(x) else float("nan")
    rows = []

    print("\n=== ① 多尺度带限相关 corr(HP_σ(OUT), HP_σ(GT))  1=结构对得上 0=只剩趋势 ===")
    print(f"{'region':8s} {'model':8s} " + " ".join(f"{'σ='+str(int(s)):>7s}" for s in SIGMAS))
    for r in REG:
        for mname in MODELS:
            vals = [nm(BAND[mname][r][s]) for s in SIGMAS]
            print(f"{r:8s} {mname:8s} " + " ".join(f"{v:7.3f}" for v in vals))
            rows.append(dict(metric="band_corr", region=r, model=mname,
                             **{f"sigma{int(s)}": v for s, v in zip(SIGMAS, vals)}))
        print()

    print("=== ② edge preservation（GT 强边缘处）===")
    print(f"{'region':8s} {'model':8s} {'|∇OUT|/|∇GT|':>14s} {'方向余弦':>10s}")
    for r in REG:
        for mname in MODELS:
            mg = nm(EDGE[mname][r]["mag"]); cs = nm(EDGE[mname][r]["cos"])
            print(f"{r:8s} {mname:8s} {mg:14.3f} {cs:10.3f}")
        print()

    print("=== ③ SSIM 三因子 ===")
    print(f"{'region':8s} {'model':8s} {'l':>7s} {'c':>7s} {'s':>7s}")
    for r in REG:
        for mname in MODELS:
            print(f"{r:8s} {mname:8s} {nm(SSIMC[mname][r]['l']):7.3f} "
                  f"{nm(SSIMC[mname][r]['c']):7.3f} {nm(SSIMC[mname][r]['s']):7.3f}")
        print()

    print("=== ④ 误差空间自相关（lag=1 / lag=8；越接近 1 说明误差越成片）===")
    print(f"{'region':8s} {'model':8s} {'lag1':>7s} {'lag8':>7s}")
    for r in REG:
        for mname in MODELS:
            print(f"{r:8s} {mname:8s} {nm(AC[mname][r][1]):7.3f} {nm(AC[mname][r][8]):7.3f}")
        print()

    with open(os.path.join(args.outdir, "clinical_metrics.csv"), "w") as f:
        f.write("region,model,sigma1,sigma2,sigma4,sigma8,sigma16,edge_mag,edge_cos,"
                "ssim_l,ssim_c,ssim_s,ac_lag1,ac_lag8\n")
        for r in REG:
            for mname in MODELS:
                f.write(",".join([r, mname] + [f"{nm(BAND[mname][r][s]):.4f}" for s in SIGMAS]
                                 + [f"{nm(EDGE[mname][r]['mag']):.4f}",
                                    f"{nm(EDGE[mname][r]['cos']):.4f}",
                                    f"{nm(SSIMC[mname][r]['l']):.4f}",
                                    f"{nm(SSIMC[mname][r]['c']):.4f}",
                                    f"{nm(SSIMC[mname][r]['s']):.4f}",
                                    f"{nm(AC[mname][r][1]):.4f}",
                                    f"{nm(AC[mname][r][8]):.4f}"]) + "\n")
    print("[save] clinical_metrics.csv")

    # ---------- 图 ----------
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    colors = {"near": "#d62728", "streak": "#fb8500", "far": "#1f77b4"}
    xs = np.arange(len(SIGMAS))
    for r in REG:
        for mname, ls in (("findnet", "--"), ("ssm", "-")):
            ys = [nm(BAND[mname][r][s]) for s in SIGMAS]
            axes[0].plot(xs, ys, ls, marker="o", ms=4, lw=1.5,
                         color=colors[r], label=f"{r} / {mname}")
    axes[0].set_xticks(xs); axes[0].set_xticklabels([f"σ={int(s)}" for s in SIGMAS])
    axes[0].set_ylabel("band-limited correlation"); axes[0].set_title("① band-limited correlation vs scale")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(color="#d9d9d9", lw=0.5, alpha=0.6); axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)

    xg = np.arange(len(REG)); bw = 0.35
    for i, mname in enumerate(MODELS):
        axes[1].bar(xg + (i - 0.5) * bw, [nm(EDGE[mname][r]["mag"]) for r in REG], bw, label=mname,
                    color="#4a4a4a" if mname == "findnet" else "#1f77b4")
    axes[1].axhline(1.0, color="#555", ls=":", lw=1)
    axes[1].set_xticks(xg); axes[1].set_xticklabels(list(REG))
    axes[1].set_ylabel("|∇OUT| / |∇GT| at GT edges"); axes[1].set_title("② edge preservation")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", color="#d9d9d9", lw=0.5, alpha=0.6); axes[1].set_axisbelow(True)

    for i, mname in enumerate(MODELS):
        axes[2].bar(xg + (i - 0.5) * bw, [nm(SSIMC[mname][r]["s"]) for r in REG], bw, label=mname,
                    color="#4a4a4a" if mname == "findnet" else "#1f77b4")
    axes[2].axhline(1.0, color="#555", ls=":", lw=1)
    axes[2].set_xticks(xg); axes[2].set_xticklabels(list(REG))
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("SSIM structure term s"); axes[2].set_title("③ SSIM structure term")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(axis="y", color="#d9d9d9", lw=0.5, alpha=0.6); axes[2].set_axisbelow(True)

    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Clinical usability beyond MAE (same held-out sequences)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(args.outdir, "clinical_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[save] clinical_metrics.png")


if __name__ == "__main__":
    main()
