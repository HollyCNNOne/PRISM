"""v22_mean_fill_test.py — 2D 先验的「均值填充」判决实验（exp013）。

要证明的一句话：
    在观测被摧毁的区域（近金属环带 / 白色条纹），FIND-Net 已经退化到「局部均值」，
    再优化也没有结构信息可用 → 必须有别的信息源（z 邻居 → SSM）。

做法：同一批 held-out 序列、同一评分域，比较五个参照
    mu_local : Gaussian(GT, sigma=8)   —— 2D 先验的贝叶斯极限（零信息解）
    findnet  : 观测端输出
    ssm      : FIND-Net + SSM
    zprior   : GT 的上一片 / 下一片     —— z 方向信息的上界
    (GT 为下界，MAE=0)

区域定义**只用推理期可见的输入**（金属 mask、LI），不用 GT（避免循环论证）：
    near   : dilate(mask, 8..NEAR_OUT px) 环带  —— 「离金属 (x,y) 越近」
    streak : |LI| > STREAK_PCT 分位，且 tissue_mask 内、不属于 near —— 「白色条纹」
    far    : 其余组织

核心判据（在 near / streak 区）：
    MAE(findnet, GT) ≈ MAE(mu_local, GT)        → 2D 已到极限，输出=均值
    MAE(zprior,  GT) << MAE(mu_local, GT)       → z 方向仍有可用信息 → 引入 SSM 有据

用法：
    python v22_mean_fill_test.py --data-dir ... --ckpt ... --outdir ... --device cuda
"""
import argparse, glob, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
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
NEAR_IN, NEAR_OUT = 8, 30      # 近金属环带（8px 是回填圈，从它外面开始测）
MU_SIGMA = 8.0                 # 局部均值参照
STREAK_PCT = 90.0              # |LI| 高分位 = 条纹


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
    """返回 near / streak / far 三个布尔域（3D: [Z,H,W]）。

    只用推理期可见：metal mask（mk）+ LI（li）+ 评分域（tm，已排除金属与回填）。
    """
    near = np.zeros_like(tm); streak = np.zeros_like(tm)
    for z in range(mk.shape[0]):
        core = mk[z] > MASK_EPS
        din = dilate2d(core, NEAR_IN)              # 含回填圈
        dout = dilate2d(core, NEAR_OUT)
        ring = dout & (~din)                       # 回填圈外侧的近场环带
        near[z] = ring & tm[z]
    a = np.abs(li)
    thr = np.percentile(a[tm], STREAK_PCT) if tm.any() else np.inf
    streak = (a > thr) & tm & (~near)
    far = tm & (~near) & (~streak)
    return near, streak, far


def inplane_hf(vol, sel):
    """面内高频能量均方：x - Gaussian_sigma3(x)。"""
    tot, cnt = 0.0, 0
    for z in range(vol.shape[0]):
        hp = vol[z] - gaussian_filter(vol[z], 3.0)
        tot += float((hp[sel[z]] ** 2).sum()); cnt += int(sel[z].sum())
    return tot / max(cnt, 1)


def mae(a, b, sel):
    d = np.abs(a - b)[sel]
    return float(d.mean()) if d.size else float("nan")


def corr(a, b, sel):
    x = a[sel].astype(np.float64); y = b[sel].astype(np.float64)
    if x.size < 10:
        return float("nan")
    x = x - x.mean(); y = y - y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else float("nan")


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
        out = trainer._final_output(res, dirty, masks, li_l)   # SSM 终输出
        # 纯 2D 参照：观测闭包在冷启动下调用一次（μ_pred = 脏图），无任何递归先验
        fn_2d = [obs(dirty[z], z) for z in range(Z)]
    m_net.train(); find_net.train()
    fn_out = torch.stack(fn_2d, dim=2)
    return out[0, 0].cpu().numpy(), fn_out[0, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--ckpt", default="/root/autodl-tmp/PRISM/training_data_astra/ssm_stage2_v22.pt")
    ap.add_argument("--patient", default="patients_head2")
    ap.add_argument("--max-seq", type=int, default=6)
    ap.add_argument("--outdir", default="/root/autodl-tmp/PRISM/v2.2/mean_fill")
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

    REGIONS = ("near", "streak", "far")
    REFS = ("baseline", "mu_local", "findnet", "ssm", "zprior")
    AUX = ("_blur_obs", "_gt_hf")
    AGG = {r: {k: [] for k in REFS + AUX} for r in REGIONS}
    NVOX = {r: [] for r in REGIONS}

    for spec in specs:
        Z = len(spec)
        gt = np.stack([read_raw(find_split(args.data_dir, s, "Target")) for s in spec])
        mk = np.stack([read_raw(find_split(args.data_dir, s, "Mask")) for s in spec])
        li = np.stack([read_raw(find_split(args.data_dir, s, "LI")) for s in spec])
        bl = np.stack([read_raw(find_split(args.data_dir, s, "Baseline")) for s in spec])
        out, fn = run_pipeline(m_net, find_net, config, trainer, args.data_dir, spec, dev)
        fn = fn[:Z]
        tm = tissue_mask(mk, gt, r=8)
        mu = np.stack([gaussian_filter(gt[z], MU_SIGMA) for z in range(Z)])
        bp = np.stack([gaussian_filter(bl[z], 4.0) for z in range(Z)])   # 观测的低通（去掉条纹噪声）
        zp = np.concatenate([gt[:1], gt[:-1]], axis=0)     # 上一片；首片用自身
        cand = {"baseline": bl, "mu_local": mu, "findnet": fn, "ssm": out,
                "zprior": zp, "_blur_obs": bp, "_gt_hf": gt}

        near, streak, far = build_regions(mk, li, tm)
        for rname, sel in (("near", near), ("streak", streak), ("far", far)):
            n = int(sel.sum())
            if n < 1000:
                continue
            NVOX[rname].append(n)
            for k, v in cand.items():
                AGG[rname][k].append(
                    (mae(v, gt, sel), inplane_hf(v, sel), corr(v, mu, sel), corr(v, gt, sel)))

    # ---------------- 汇总 ----------------
    rows = []
    print("\n=== 区域 × 参照：MAE(HU) / 面内HF / corr(·,μ_local) / corr(·,GT) ===")
    print(f"{'region':8s} {'ref':12s} {'MAE':>8s} {'HF':>9s} {'c_mu':>7s} {'c_gt':>7s} {'voxels':>10s}")
    for rname in REGIONS:
        if not NVOX[rname]:
            continue
        n = int(np.mean(NVOX[rname]))
        for k in REFS + AUX:
            arr = np.array(AGG[rname][k], dtype=float)
            m_ = float(np.nanmean(arr[:, 0]))
            h_ = float(np.nanmean(arr[:, 1]))
            cm = float(np.nanmean(arr[:, 2]))
            cg = float(np.nanmean(arr[:, 3]))
            rows.append(dict(region=rname, ref=k, mae=m_, hf=h_, corr_mu=cm, corr_gt=cg, n=n))
            print(f"{rname:8s} {k:12s} {m_:8.1f} {h_:9.1f} {cm:7.3f} {cg:7.3f} {n:10d}")
        print()

    # ---- 核心判据表：MAE 比值 ----
    print("=== 核心判据（相对 μ_local 的 MAE 比，<1 表示优于均值填充）===")
    print(f"{'region':8s} {'FINDNet/μ':>10s} {'SSM/μ':>8s} {'zprior/μ':>9s}")
    crit = {}
    for rname in REGIONS:
        if not NVOX[rname]:
            continue
        crit[rname] = {k: float(np.nanmean([a[0] for a in AGG[rname][k]])) for k in REFS}
        m_mu = crit[rname]["mu_local"]
        print(f"{rname:8s} {crit[rname]['findnet']/m_mu:10.3f} "
              f"{crit[rname]['ssm']/m_mu:8.3f} {crit[rname]['zprior']/m_mu:9.3f}")

    with open(os.path.join(args.outdir, "mean_fill.csv"), "w") as f:
        f.write("region,ref,mae_HU,inplane_hf,corr_mu_local,corr_gt,voxels\n")
        for r in rows:
            f.write(f"{r['region']},{r['ref']},{r['mae']:.3f},{r['hf']:.1f},"
                    f"{r['corr_mu']:.4f},{r['corr_gt']:.4f},{r['n']}\n")
    print("[save] mean_fill.csv")

    # ---- 图：三个区域的 MAE 柱状 + 相对均值比 ----
    # 作图包 try/except：数字已落盘，图挂了不能连累结果
    try:
        plt.rcParams.update({"font.family": "serif",
                             "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                             "font.size": 10})
        regs = [r for r in REGIONS if r in crit]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        xs = np.arange(len(regs)); bw = 0.2
        labels = [("mu_local", "mu_local (2D limit)", "#999999"),
                  ("findnet", "FIND-Net", "#d62728"),
                  ("ssm", "FIND-Net+SSM", "#1f77b4"),
                  ("zprior", "GT z-neighbour (upper bound)", "#2ca02c")]
        for i, (k, lab, c) in enumerate(labels):
            vals = [crit[r][k] for r in regs]
            axes[0].bar(xs + (i - 1.5) * bw, vals, bw, label=lab, color=c)
            rr = [crit[r][k] / crit[r]["mu_local"] for r in regs]
            axes[1].bar(xs + (i - 1.5) * bw, rr, bw, label=lab, color=c)
        for ax in axes:
            ax.set_xticks(xs); ax.set_xticklabels(regs)
            ax.grid(axis="y", color="#d9d9d9", lw=0.5, alpha=0.6); ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[0].set_ylabel("MAE vs GT (HU)"); axes[0].set_title("Absolute error by region")
        axes[1].axhline(1.0, color="#555", ls=":", lw=1)
        axes[1].set_ylabel("MAE / MAE(mu_local)"); axes[1].set_title("Relative to mean-fill limit")
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Mean-fill test: is 2D prior exhausted where observation is destroyed?",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(os.path.join(args.outdir, "mean_fill.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("[save] mean_fill.png")
    except Exception as e:
        print(f"[WARN] 作图失败（数字已保存）: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
