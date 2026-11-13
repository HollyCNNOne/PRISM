#!/usr/bin/env python
"""gen_directional_obs.py —— 生成方向滤波观测 x̂_dir（探针第 2 步，数据侧一次性落盘）。

对每个样本：结构张量提取条纹方向 → 金属邻域内 coherence 加权双倍角平均得「样本级主方向」
→ 沿垂直条纹方向各向异性低通（抹条纹、保平行组织）→ masked（金属邻域 ∩ 阴影区 r<9）
→ 落盘 x̂_dir 到 Baseline_dir/。

为什么是「样本级主方向」而不是逐体素 steerable：
  第 1 步 steerable(8方向最近邻) vs 全局主方向 在 5 个真实金属样本上对照，rmse_near 差异
  ≤1.6%（统计噪声内），因为线圈小、条纹方向在 20px 金属邻域内近均匀。故全局主方向已够，
  省掉 8× FFT，只用 1× FFT。若未来 giant 线圈弯曲明显，可再升级逐体素。

方向滤波是观测的确定性预处理（冻结输入，不参与梯度），数据侧预计算落盘、训练循环零额外
成本。x̂_dir 只在「金属邻域且阴影区」替换观测，充分区/远离金属的原样保护（rmse_far 精确
不变）。

用法（服务器，GPU 训练 env 的 numpy 即可）：
    python gen_directional_obs.py --data-dir /root/autodl-tmp/PRISM/training_data \
        --n-samples 3            # 先验证写盘
    python gen_directional_obs.py --data-dir /root/autodl-tmp/PRISM/training_data \
        --all                    # 全量落盘
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stripe_direction_probe import (  # noqa: E402
    structure_tensor, anisotropic_smooth, separable_smooth)

H = W = 512


def read_raw(path: str, h: int = H, w: int = W) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(h, w)


def write_raw(path: str, arr: np.ndarray) -> None:
    arr.astype(np.float32).tofile(path)


def metal_neighbor_mask(x_M0: np.ndarray, radius: float = 20.0) -> np.ndarray:
    """距离金属 < radius 的体素 = 1（高斯膨胀近似，无 scipy 依赖）。"""
    m = (x_M0 > 0).astype(np.float64)
    m_blur = separable_smooth(m, radius / 3.0)
    return (m_blur > 0.01).astype(np.float64)


def directional_obs(x_hat, x_M0, d_j, d_ref, sigma_par=1.0, sigma_perp=4.0):
    """方向滤波观测：θ 提取 + 样本级主方向各向异性低通 + masked 替换。

    返回 (x_dir, theta_main, masks)。masks 含 partial/complete 分带，供统计解读：
    B8 卡尔曼里完全阴影(r<1/9) K=1 丢弃观测、部分阴影(1/9≤r<9) K→0 全信观测，故方向滤波
    的 carry 收益只在部分阴影；完全阴影观测被丢，滤波无害但无意义。分带统计把两者拆开。
    """
    theta, coh = structure_tensor(x_hat, sigma=1.5)
    r = d_j / d_ref
    near = metal_neighbor_mask(x_M0, radius=20.0) * (r < 9.0).astype(np.float64)
    partial = metal_neighbor_mask(x_M0, radius=20.0) * \
        ((r >= 1.0 / 9.0) & (r < 9.0)).astype(np.float64)
    complete = metal_neighbor_mask(x_M0, radius=20.0) * (r < 1.0 / 9.0).astype(np.float64)

    # 样本级主方向：金属邻域阴影区内 coherence 加权双倍角平均（无 Target 泄漏）
    m = (near > 0.5) & (coh > 0.3)
    if m.sum() < 100:
        # 阴影/金属邻域太小，无可靠方向 → 不滤波，原样返回
        return x_hat.copy(), float("nan"), {"near": near, "partial": partial,
                                            "complete": complete}
    th = theta[m]
    theta_main = 0.5 * np.arctan2(float(np.mean(np.sin(2.0 * th))),
                                  float(np.mean(np.cos(2.0 * th)))) % np.pi

    x_dir = anisotropic_smooth(x_hat, theta_main, sigma_par, sigma_perp)
    out = near * x_dir + (1.0 - near) * x_hat
    return out, theta_main, {"near": near, "partial": partial, "complete": complete}


def rmse(a, b, sel=None):
    if sel is not None:
        a, b = a[sel], b[sel]
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def list_samples(phys_dir, frac_lo=0.0, frac_hi=1.0):
    """列所有有 Phys x_M0 的样本，按 sample_id 排序（全量落盘用）。"""
    ids = sorted(
        os.path.basename(f).replace("_x_M0.npy", "")
        for f in glob.glob(os.path.join(phys_dir, "*_x_M0.npy")))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data")
    ap.add_argument("--out-dir", default=None, help="落盘目录；缺省 train/Baseline_dir")
    ap.add_argument("--n-samples", type=int, default=0,
                    help="只处理前 N 个（验证写盘）；0=全部")
    ap.add_argument("--all", action="store_true", help="全量（等价 --n-samples 0）")
    ap.add_argument("--sigma-par", type=float, default=1.0)
    ap.add_argument("--sigma-perp", type=float, default=4.0)
    args = ap.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    phys_dir = os.path.join(train_dir, "Phys")
    base_dir = os.path.join(train_dir, "Baseline")
    tgt_dir = os.path.join(train_dir, "Target")
    out_dir = args.out_dir or os.path.join(train_dir, "Baseline_dir")
    os.makedirs(out_dir, exist_ok=True)

    d_ref_path = os.path.join(args.data_dir, "d_ref_ho_patients_head2.npy")
    if os.path.exists(d_ref_path):
        d_ref = float(np.load(d_ref_path)[0])
    else:
        d_ref = float(np.load(os.path.join(args.data_dir, "d_ref.npy"))[0])
    print(f"[d_ref] {d_ref:.6g}", flush=True)

    samples = list_samples(phys_dir)
    print(f"[find] {len(samples)} samples with x_M0", flush=True)
    if not args.all and args.n_samples > 0:
        samples = samples[: args.n_samples]

    all_stats = []
    for i, sid in enumerate(samples):
        t0 = time.time()
        x_hat = read_raw(os.path.join(base_dir, f"{sid}.raw"))
        target = read_raw(os.path.join(tgt_dir, f"{sid}.raw"))
        x_M0 = np.load(os.path.join(phys_dir, f"{sid}_x_M0.npy"))[0]
        d_j = np.load(os.path.join(phys_dir, f"{sid}_d_j.npy"))[0]

        x_dir, theta_main, masks = directional_obs(
            x_hat, x_M0, d_j, d_ref, args.sigma_par, args.sigma_perp)

        near = masks["near"] > 0.5
        partial = masks["partial"] > 0.5
        complete = masks["complete"] > 0.5
        far = ~near
        stats = {
            "sample_id": sid,
            "metal_frac": float((x_M0 > 0).mean()),
            "theta_main_deg": float(np.rad2deg(theta_main)),
            "near_frac": float(near.mean()),
            "partial_frac": float(partial.mean()),
            "complete_frac": float(complete.mean()),
            # 部分阴影 = 滤波 carry 收益区（观测被全信）
            "rmse_before_partial": rmse(x_hat, target, partial),
            "rmse_after_partial": rmse(x_dir, target, partial),
            # 完全阴影 = 观测被丢弃，滤波无害无意义（仅示伪影多严重）
            "rmse_before_complete": rmse(x_hat, target, complete),
            "rmse_after_complete": rmse(x_dir, target, complete),
            # 远离金属 = 必须精确不变
            "rmse_before_far": rmse(x_hat, target, far),
            "rmse_after_far": rmse(x_dir, target, far),
        }

        out_path = os.path.join(out_dir, f"{sid}.raw")
        write_raw(out_path, x_dir)
        stats["out_path"] = out_path

        all_stats.append(stats)
        if i % 100 == 0 or i == len(samples) - 1:
            print(f"[proc {i+1}/{len(samples)}] {sid} {time.time()-t0:.2f}s "
                  f"partial {stats['rmse_before_partial']:.0f}->{stats['rmse_after_partial']:.0f} "
                  f"far_delta {stats['rmse_after_far']-stats['rmse_before_far']:.2e}",
                  flush=True)

    with open(os.path.join(out_dir, "gen_stats.json"), "w") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"[done] {len(all_stats)} samples -> {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
