#!/usr/bin/env python
"""dsp_gt_vs_baseline.py —— GT vs Baseline 的 z 轴频域数值实验（只出数值，不画图）。

目的：搞清三个关系
  (1) 同一完备切片组内，GT 与 Baseline 的 z 轴频域差（伪影的高频特征有多强）；
  (2) 不同完备切片组(case_category) / 重建核(kernel) / Z 长度 之间，频域趋势/波动范围是否一致；
  (3) 低频 vs 组织高频 vs 高高频 三层的分离是否显著（短序列下是否粗）。

信号定义（固定 (x,y) 小块，沿 z 追踪——Z-FSP 的正确单位）：
  metal  块：金属掩膜质心 (cy,cx)，半宽 P=8，信号 = 块内 MAX HU（抓金属峰 0→3000→0 阶跃）。
  streak 块：质心右侧 (cy, cx+35)，半宽 P=12，信号 = 块内 MEAN HU（条纹带）。
  tissue 块：非金属、离金属≥50px、GT HU∈[-150,250]（脑组织），取 d_j 最大处，半宽 P=12，MEAN HU。

每块算 GT 与 Baseline 的 z 信号 → 去 DC → rfft → 频域能量：
  dc     = mean（整体水平，金属区 GT vs Base 的 HU 差也在这里体现）
  ac     = sum_{f>0} |FFT|²（总高频能量）
  ac_low = 第一个 AC bin（f=1/N，低频）
  ac_hi  = 其余 AC bin（f>=2/N，高频）
  ratio  = Baseline_ac / GT_ac

输出：per-sequence 全表（CSV）+ 按 region × case_category / kernel / Z 分层的汇总表。
"""
import glob
import json
import os
import csv

import numpy as np

D = "/root/autodl-tmp/PRISM/training_data_astra/train"
META = "/root/autodl-tmp/PRISM/training_data_astra/meta"
D_REF = 4.10023e+06


def rd(sid, sub):
    return np.fromfile(os.path.join(D, sub, f"{sid}.raw"), np.float32).reshape(512, 512)


def sample_idx(sid):
    return int(sid.rsplit("_", 1)[1])


def fft_energy(sig):
    """去 DC 后 rfft，返回 (dc, ac, ac_low, ac_hi)。"""
    dc = float(sig.mean())
    F = np.abs(np.fft.rfft(sig - dc))
    ac = float((F[1:] ** 2).sum())
    ac_low = float(F[1] ** 2) if len(F) > 1 else 0.0
    ac_hi = float((F[2:] ** 2).sum()) if len(F) > 2 else 0.0
    return dc, ac, ac_low, ac_hi


def _c(v, P):
    return int(min(max(v, P), 511 - P))


def patch_max(vol, cy, cx, P):
    return np.array([vol[z, cy - P:cy + P + 1, cx - P:cx + P + 1].max() for z in range(vol.shape[0])])


def patch_mean(vol, cy, cx, P):
    return np.array([vol[z, cy - P:cy + P + 1, cx - P:cx + P + 1].mean() for z in range(vol.shape[0])])


def main():
    by_seq = {}
    for f in glob.glob(os.path.join(META, "*.json")):
        m = json.load(open(f))
        if m.get("split") == "train":
            by_seq.setdefault(m["sequence_id"], []).append(m)

    rows = []
    for seq_id, metas in by_seq.items():
        if len(metas) < 4:
            continue
        metas = sorted(metas, key=lambda m: sample_idx(m["sample_id"]))
        sids = [m["sample_id"] for m in metas]
        Z = len(sids)
        kernel = metas[0]["kernel"]
        cc = metas[0]["case_category"]

        msk = np.stack([rd(s, "Mask") for s in sids])
        if (msk > 0.5).sum() < 300:
            continue
        gt = np.stack([rd(s, "Target") for s in sids])
        base = np.stack([rd(s, "Baseline") for s in sids])
        dj = np.stack([np.load(os.path.join(D, "Phys", f"{s}_d_j.npy")).reshape(512, 512)
                       for s in sids])

        # 中片金属质心
        zi = Z // 2
        ys, xs = np.nonzero(msk[zi] > 0.5)
        cy, cx = int(ys.mean()), int(xs.mean())

        # tissue 中心：非金属、离金属>=50px、GT HU∈[-150,250]，取 d_j 最大
        yy, xx = np.mgrid[0:512, 0:512]
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mid_gt = gt[zi]
        mid_dj = dj[zi]
        cand = (msk[zi] < 0.5) & (dist >= 50) & (mid_gt >= -150) & (mid_gt <= 250)
        if cand.sum() > 50:
            tc = np.unravel_index(int(np.argmax(np.where(cand, mid_dj, -1.0))), mid_dj.shape)
            ty, tx = _c(int(tc[0]), 12), _c(int(tc[1]), 12)
        else:
            ty, tx = _c(cy - 60, 12), _c(cx - 40, 12)

        patches = {
            "metal": (patch_max, cy, cx, 8),
            "streak": (patch_mean, cy, _c(cx + 35, 12), 12),
            "tissue": (patch_mean, ty, tx, 12),
        }
        for name, (fn, py, px, P) in patches.items():
            g = fn(gt, py, px, P)
            b = fn(base, py, px, P)
            gdc, gac, glo, ghi = fft_energy(g)
            bdc, bac, blo, bhi = fft_energy(b)
            rows.append({
                "seq": int(seq_id), "Z": Z, "kernel": kernel, "case_category": cc,
                "region": name, "gt_dc": gdc, "gt_ac": gac, "gt_ac_low": glo, "gt_ac_hi": ghi,
                "base_dc": bdc, "base_ac": bac, "base_ac_low": blo, "base_ac_hi": bhi,
                "ratio_ac": bac / (gac + 1e-9),
            })

    # ---- 全表落盘 ----
    out_csv = "/root/autodl-tmp/PRISM/training_data_astra/viz_v12/dsp_gt_baseline_full.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    keys = ["seq", "Z", "kernel", "case_category", "region", "gt_dc", "gt_ac", "gt_ac_low",
            "gt_ac_hi", "base_dc", "base_ac", "base_ac_low", "base_ac_hi", "ratio_ac"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[save] {out_csv}  ({len(rows)} rows)", flush=True)

    def summ(sel):
        if sel.sum() == 0:
            return None
        d = [r for r, s in zip(rows, sel) if s]
        def agg(k):
            v = np.array([r[k] for r in d])
            return (np.median(v), np.percentile(v, 25), np.percentile(v, 75))
        return {
            "n": len(d),
            "gt_ac": agg("gt_ac"), "base_ac": agg("base_ac"), "ratio": agg("ratio_ac"),
            "gt_dc": agg("gt_dc"), "base_dc": agg("base_dc"),
            "gt_ac_hi": agg("gt_ac_hi"), "base_ac_hi": agg("base_ac_hi"),
        }

    print("\n===== 按 region（总体）=====", flush=True)
    for name in ["metal", "streak", "tissue"]:
        s = summ(np.array([r["region"] == name for r in rows]))
        if s:
            print(f"  {name:7s} n={s['n']:3d}  gt_dc={s['gt_dc'][0]:8.0f}  base_dc={s['base_dc'][0]:8.0f}  "
                  f"gt_ac={s['gt_ac'][0]:9.0f}  base_ac={s['base_ac'][0]:9.0f}  ratio_ac={s['ratio'][0]:6.2f}  "
                  f"gt_ac_hi={s['gt_ac_hi'][0]:8.0f}  base_ac_hi={s['base_ac_hi'][0]:8.0f}", flush=True)

    print("\n===== metal 区 × case_category（完备度）=====", flush=True)
    for cc in sorted(set(r["case_category"] for r in rows)):
        s = summ(np.array([r["region"] == "metal" and r["case_category"] == cc for r in rows]))
        if s:
            print(f"  {cc:16s} n={s['n']:3d}  gt_ac={s['gt_ac'][0]:9.0f}  base_ac={s['base_ac'][0]:9.0f}  "
                  f"ratio={s['ratio'][0]:6.2f}  base_dc={s['base_dc'][0]:8.0f}", flush=True)

    print("\n===== tissue 区 × case_category（完备度）=====", flush=True)
    for cc in sorted(set(r["case_category"] for r in rows)):
        s = summ(np.array([r["region"] == "tissue" and r["case_category"] == cc for r in rows]))
        if s:
            print(f"  {cc:16s} n={s['n']:3d}  gt_ac={s['gt_ac'][0]:9.0f}  base_ac={s['base_ac'][0]:9.0f}  "
                  f"ratio={s['ratio'][0]:6.2f}", flush=True)

    print("\n===== metal 区 × kernel =====", flush=True)
    for kk in sorted(set(r["kernel"] for r in rows)):
        s = summ(np.array([r["region"] == "metal" and r["kernel"] == kk for r in rows]))
        if s:
            print(f"  {kk:14s} n={s['n']:3d}  gt_ac={s['gt_ac'][0]:9.0f}  base_ac={s['base_ac'][0]:9.0f}  "
                  f"ratio={s['ratio'][0]:6.2f}", flush=True)

    print("\n===== metal 区 × Z 长度（短/长）=====", flush=True)
    for lab, cond in [("Z<=6", lambda r: r["Z"] <= 6), ("Z>=7", lambda r: r["Z"] >= 7)]:
        s = summ(np.array([r["region"] == "metal" and cond(r) for r in rows]))
        if s:
            print(f"  {lab:6s} n={s['n']:3d}  gt_ac={s['gt_ac'][0]:9.0f}  base_ac={s['base_ac'][0]:9.0f}  "
                  f"ratio={s['ratio'][0]:6.2f}", flush=True)


if __name__ == "__main__":
    main()
