"""v22_observation_ceiling.py — 观测端的高频天花板（线 1 的证据，纯 numpy，不用网络）。

问题：近环（金属外 8–30px）里，2D 后处理到底还有没有高频信息可用？

做法：直接在数据上算「观测（Baseline）的高频」与「真值的高频」的相关
    corr( HP_σ(Baseline), HP_σ(GT) )
    corr( HP_σ(blur(Baseline,4)), HP_σ(GT) )     # 观测能做到的最好低通
    σ = 1, 2, 4, 8, 16

语义（必须写准）：
    该相关 = 以 HP_σ(Baseline) 为唯一输入的**最优线性估计器**对 HP_σ(GT) 的解释力。
    Wiener 滤波达到的就是这个值。深度网络原则上可用非线性结构超过它，
    所以它是「线性天花板」，不是绝对上界 —— 但它足以说明观测端还剩多少油水。

对照：GT 上一片（z 邻居）作为另一个信息源，见 exp013。

区域定义与 exp014 完全一致（只用推理期可见的 metal mask / LI，不用 GT）。
"""
import argparse, glob, json, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

MASK_EPS = 1e-3
NEAR_IN, NEAR_OUT = 8, 30
STREAK_PCT = 90.0
SIGMAS = (1.0, 2.0, 4.0, 8.0, 16.0)


def read_raw(p):
    return np.fromfile(p, np.float32).reshape(512, 512)


def find_split(d, sid, sub):
    for sp in ("train", "val"):
        p = os.path.join(d, sp, sub, f"{sid}.raw")
        if os.path.exists(p):
            return p
    return None


def dilate2d(b, r):
    if r <= 0:
        return b
    return maximum_filter(b, size=(2 * r + 1, 2 * r + 1), mode="constant", cval=False)


def tissue_mask_np(mask, gt, r=8, metal_eps=1e-3, air_hu=-500.0):
    """与 prism/eval/domain.py::tissue_mask 同义（本脚本不依赖 torch）。"""
    metal = mask > metal_eps
    if r > 0:
        bf = np.zeros_like(metal)
        for z in range(metal.shape[0]):
            bf[z] = dilate2d(metal[z], r)
    else:
        bf = metal
    return (~bf) & (gt > air_hu)


def build_regions(mk, li, tm):
    near = np.zeros_like(tm)
    for z in range(mk.shape[0]):
        core = mk[z] > MASK_EPS
        near[z] = (dilate2d(core, NEAR_OUT) & (~dilate2d(core, NEAR_IN))) & tm[z]
    a = np.abs(li)
    thr = np.percentile(a[tm], STREAK_PCT) if tm.any() else np.inf
    streak = (a > thr) & tm & (~near)
    far = tm & (~near) & (~streak)
    return {"near": near, "streak": streak, "far": far}


def band_corr(x, y, sel, sigma):
    """分区内的 HP_σ 相关（逐片算再平均）。"""
    vals = []
    for z in range(x.shape[0]):
        s = sel[z]
        if s.sum() < 50:
            continue
        hx = x[z] - gaussian_filter(x[z], sigma)
        hy = y[z] - gaussian_filter(y[z], sigma)
        a = hx[s].astype(np.float64); b = hy[s].astype(np.float64)
        if a.size < 50:
            continue
        a = a - a.mean(); b = b - b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        if den <= 0:
            continue
        vals.append(float((a * b).sum() / den))
    return float(np.mean(vals)) if vals else float("nan")


def hf_energy(x, sel, sigma=3.0):
    tot, cnt = 0.0, 0
    for z in range(x.shape[0]):
        hp = x[z] - gaussian_filter(x[z], sigma)
        tot += float((hp[sel[z]] ** 2).sum()); cnt += int(sel[z].sum())
    return tot / max(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--patient", default="patients_head2")
    ap.add_argument("--max-seq", type=int, default=6)
    ap.add_argument("--outdir", default="/root/autodl-tmp/PRISM/v2.2/ceiling")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

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
    SRC = ("baseline", "blur4", "zprior")
    AGG = {r: {k: {s: [] for s in SIGMAS} for k in SRC} for r in REG}
    HF = {r: {"gt": [], "baseline": []} for r in REG}
    NV = {r: [] for r in REG}

    for spec in specs:
        Z = len(spec)
        gt = np.stack([read_raw(find_split(args.data_dir, s, "Target")) for s in spec])
        mk = np.stack([read_raw(find_split(args.data_dir, s, "Mask")) for s in spec])
        li = np.stack([read_raw(find_split(args.data_dir, s, "LI")) for s in spec])
        bl = np.stack([read_raw(find_split(args.data_dir, s, "Baseline")) for s in spec])
        tm = tissue_mask_np(mk, gt, r=8)
        bl4 = np.stack([gaussian_filter(bl[z], 4.0) for z in range(Z)])
        zp = np.concatenate([gt[:1], gt[:-1]], axis=0)
        cand = {"baseline": bl, "blur4": bl4, "zprior": zp}
        regions = build_regions(mk, li, tm)
        for rn, sel in regions.items():
            if sel.sum() < 2000:
                continue
            NV[rn].append(int(sel.sum()))
            HF[rn]["gt"].append(hf_energy(gt, sel))
            HF[rn]["baseline"].append(hf_energy(bl, sel))
            for k in SRC:
                for s in SIGMAS:
                    AGG[rn][k][s].append(band_corr(cand[k], gt, sel, s))
        print(f"  seq {spec[0]} Z={Z} done", flush=True)

    nm = lambda v: float(np.nanmean(v)) if len(v) else float("nan")
    rows = []
    print("\n=== 观测端高频天花板 corr(HP_σ(src), HP_σ(GT)) ===")
    print(f"{'region':8s} {'src':10s} " + " ".join(f"{'σ='+str(int(s)):>7s}" for s in SIGMAS))
    for r in REG:
        if not NV[r]:
            continue
        for k in SRC:
            vals = [nm(AGG[r][k][s]) for s in SIGMAS]
            print(f"{r:8s} {k:10s} " + " ".join(f"{v:7.3f}" for v in vals))
            rows.append((r, k, vals, int(np.mean(NV[r]))))
        print()

    print("=== 面内高频能量（σ=3 高通，同区同口径）===")
    print(f"{'region':8s} {'GT':>12s} {'Baseline':>12s} {'BL/GT':>8s} {'voxels':>10s}")
    for r in REG:
        if not NV[r]:
            continue
        g = nm(HF[r]["gt"]); b = nm(HF[r]["baseline"])
        print(f"{r:8s} {g:12.1f} {b:12.1f} {b/max(g,1e-9):8.2f} {int(np.mean(NV[r])):10d}")

    with open(os.path.join(args.outdir, "observation_ceiling.csv"), "w") as f:
        f.write("region,source,sigma1,sigma2,sigma4,sigma8,sigma16,voxels\n")
        for r, k, vals, n in rows:
            f.write(f"{r},{k}," + ",".join(f"{v:.4f}" for v in vals) + f",{n}\n")
        for r in REG:
            if not NV[r]:
                continue
            f.write(f"{r},hf_gt,{nm(HF[r]['gt']):.1f},,,,\n")
            f.write(f"{r},hf_baseline,{nm(HF[r]['baseline']):.1f},,,,\n")
    print("[save]", os.path.join(args.outdir, "observation_ceiling.csv"))


if __name__ == "__main__":
    main()
