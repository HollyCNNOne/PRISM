#!/usr/bin/env python
"""characterize_residual.py —— 拆解金属伪影残差 δ = x̂ − target 的结构（探针诊断，不落地）。

回答：部分阴影（观测被全信、是 carry 偏置的来源）里的残差到底长什么样？
  (a) 高频条纹（垂直金属方向）→ 方向滤波本该管用，是我实现错了；
  (b) 低频阴影/束硬化 → 方向滤波从根上找错频带，carry 是低频 bias，不是条纹；
  (c) 各向同性噪声 → 方向性根本无关。

量（在部分阴影 / 完全阴影两个区域分别算，target 仅用于诊断，不落地）：
  - rmse_total：残差总幅度
  - frac_low：低通(σ=8)残差的能量占比 —— 「阴影/束硬化」部分
  - frac_high：高通残差的能量占比 —— 「条纹+组织边缘」部分
  - aniso：高通残差沿「垂直条纹方向」梯度能量 / 「平行条纹方向」梯度能量 —— >1=条纹状
           （条纹的灰度沿平行方向近常数 → 平行梯度小 → aniso 大）
  - aniso_total：残差整体（含低频）的同一比值

判断表：
  部分阴影 frac_high 小 & frac_low 大  → carry 是低频 bias，方向滤波找错频带（→ bias 校正头）
  部分阴影 frac_high 大 & aniso 大     → carry 是条纹，方向概念对，我实现错了（→ 换 innovation 滤波）
  部分阴影 aniso≈1                     → 方向性无关

用法（服务器）：
    python characterize_residual.py --data-dir /root/autodl-tmp/PRISM/training_data --n 30
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_directional_obs import read_raw, metal_neighbor_mask
from stripe_direction_probe import structure_tensor, separable_smooth, anisotropic_smooth


def characterize(x_hat, target, x_M0, d_j, d_ref):
    r = d_j / d_ref
    mn = metal_neighbor_mask(x_M0, radius=20.0)
    partial = (mn > 0.5) & ((r >= 1.0 / 9.0) & (r < 9.0))
    complete = (mn > 0.5) & (r < 1.0 / 9.0)

    theta, coh = structure_tensor(x_hat, sigma=1.5)

    out = {}
    for name, sel in (("partial", partial), ("complete", complete)):
        if sel.sum() < 50:
            out[name] = {"n": int(sel.sum())}
            continue
        e = x_hat - target
        # 区域主方向（coherence 加权双倍角，来自观测 x_hat 的结构张量）
        m = sel & (coh > 0.3)
        if m.sum() >= 100:
            th = theta[m]
            th_main = 0.5 * np.arctan2(float(np.mean(np.sin(2 * th))),
                                       float(np.mean(np.cos(2 * th)))) % np.pi
            th_main_deg = float(np.rad2deg(th_main))
            perp = th_main_deg + 90.0
        else:
            th_main_deg = float("nan")
            perp = float("nan")

        def rms(img):
            return float(np.sqrt(np.mean(img[sel] ** 2)))

        e_before = rms(e)                                # 滤波前残差
        # 方向滤波 D 的精确分解：e' = D(x̂)-target = D(e) + (D(target)-target)
        D_e = anisotropic_smooth(e, th_main, 1.0, 4.0) if not np.isnan(perp) else e
        D_tgt = anisotropic_smooth(target, th_main, 1.0, 4.0) if not np.isnan(perp) else target
        res_removed = rms(D_e)                          # 滤波后残差剩多少（<before=滤掉了一部分）
        tissue_damage = rms(D_tgt - target)             # 滤波对干净组织的损伤

        # 各向异性（干净版）：梯度在整图上算，再按 sel 取平均（不零化，避免边界伪影）
        if not np.isnan(perp):
            gy, gx = np.gradient(e.astype(np.float64))
            a_perp = np.deg2rad(perp)
            a_par = np.deg2rad(th_main_deg)
            e_perp = float(np.mean((gx * np.cos(a_perp) + gy * np.sin(a_perp)) ** 2))
            e_par = float(np.mean((gx * np.cos(a_par) + gy * np.sin(a_par)) ** 2))
            aniso = e_perp / max(e_par, 1e-12)
        else:
            aniso = float("nan")

        out[name] = {
            "n": int(sel.sum()),
            "rmse_before": e_before,
            "res_removed": res_removed,                 # <before ⇒ 方向滤波确实滤掉了残差
            "tissue_damage": tissue_damage,             # 方向滤波损伤的组织量
            "aniso": aniso,                             # 残差垂直/平行梯度能量比（>1=条纹状）
            "theta_main_deg": th_main_deg,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data")
    ap.add_argument("--n", type=int, default=30, help="取前 N 个有意义金属样本")
    args = ap.parse_args()

    train = os.path.join(args.data_dir, "train")
    phys = os.path.join(train, "Phys")
    base = os.path.join(train, "Baseline")
    tgt = os.path.join(train, "Target")
    d_ref = float(np.load(os.path.join(args.data_dir, "d_ref_ho_patients_head2.npy"))[0])

    # 有意义金属样本（0.5%~15%），均匀跨尺寸取 N 个
    ids = []
    for f in glob.glob(os.path.join(phys, "*_x_M0.npy")):
        a = np.load(f)
        frac = float((a > 0).mean())
        if 0.005 <= frac <= 0.15:
            ids.append((frac, os.path.basename(f).replace("_x_M0.npy", "")))
    ids.sort()
    step = max(1, len(ids) // args.n)
    sub = ids[::step][: args.n]

    # 聚合（能量加权：均方误差按体素数加权）
    agg = {"partial": {}, "complete": {}}
    for zone in ("partial", "complete"):
        agg[zone] = {"n": 0, "e_before": 0.0, "res_removed": 0.0,
                     "tissue_damage": 0.0, "aniso_w": 0.0}

    recs = []
    for frac, sid in sub:
        xh = read_raw(os.path.join(base, sid + ".raw"))
        tg = read_raw(os.path.join(tgt, sid + ".raw"))
        xm = np.load(os.path.join(phys, sid + "_x_M0.npy"))[0]
        dj = np.load(os.path.join(phys, sid + "_d_j.npy"))[0]
        c = characterize(xh, tg, xm, dj, d_ref)
        c["sample_id"] = sid
        c["metal_frac"] = frac
        recs.append(c)
        for zone in ("partial", "complete"):
            z = c[zone]
            if "rmse_before" not in z or z.get("n", 0) < 50:
                continue
            a = agg[zone]
            a["n"] += z["n"]
            a["e_before"] += z["rmse_before"] ** 2 * z["n"]
            a["res_removed"] += z["res_removed"] ** 2 * z["n"]
            a["tissue_damage"] += z["tissue_damage"] ** 2 * z["n"]
            if not np.isnan(z["aniso"]):
                a["aniso_w"] += z["aniso"] * z["n"]

    print("=== 聚合（能量加权，n=体素数）===", flush=True)
    for zone in ("partial", "complete"):
        a = agg[zone]
        if a["n"] == 0:
            print(f"[{zone}] 无体素", flush=True)
            continue
        before = np.sqrt(a["e_before"] / a["n"])
        removed = np.sqrt(a["res_removed"] / a["n"])
        damage = np.sqrt(a["tissue_damage"] / a["n"])
        aniso = a["aniso_w"] / a["n"]
        print(f"[{zone}] n={a['n']}  rmse_before={before:.0f}  "
              f"res_removed(滤波后残差)={removed:.0f}  tissue_damage={damage:.0f}  "
              f"aniso={aniso:.2f}", flush=True)

    with open(os.path.join(os.path.dirname(args.data_dir), "resid_char.json"), "w") as f:
        json.dump(recs, f, indent=2, ensure_ascii=False)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
