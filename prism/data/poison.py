"""poison —— 训练期「假阳性注入增强」（Data Poisoning）。

动机（需求三 + 物理一致性）：仿真的 x_M0 已知精确（纯线圈），模型训练时永远见不到
假阳性金属；而真实数据退回到粗糙 HU 阈值后（metal_threshold.raw_metal_mask），x_M0
必然混入牙齿/骨/假牙。为了不让训练分布与真实分布脱节，我们在训练批次里**故意**把底
图中的真牙（HU>teeth_hu）强行标成金属、拼接到 x_M0 输入通道，逼 M_μ/SSM 依靠高 d_j
（牙齿无光子饥饿）与 z 轴连续性，在内部把这段假金属过滤掉，仍输出干净组织。

注意：本模块只改 x_M0 输入通道（投毒），不动 Target/X_clean（监督仍指向干净组织）。
真线圈体素不受影响——用 np.maximum 覆盖（牙齿/线圈解剖上不重叠），等效「拼接」。

本模块纯 numpy（数据加载侧，在 collate 之前调用），本机可自检；不破坏 3D 连读接口：
输入输出同形 [Z,1,H,W]，逐 z 片操作、跨片用同一空间掩膜（z 连续）。
"""

from __future__ import annotations

import numpy as np

# 真牙/致密骨的 HU 上限量级；Pt-W 线圈远高于此（peak ~35000）。默认与 L_pc 的
# pc_metal_hu_thr 区分开：这里取 2500 抓「真牙」，不抓线圈。
TEETH_HU_DEFAULT = 2500.0


def inject_false_metal(x_M0, X_clean, metal_mu=None, teeth_hu: float = TEETH_HU_DEFAULT,
                       prob: float = 0.5, frac: float = 1.0, seed: int = None):
    """把底图中的真牙/致密骨（HU>teeth_hu）随机标成金属，拼接到 x_M0 通道。

    Parameters
    ----------
    x_M0: [Z,1,H,W] 或 [1,H,W] 真实金属先验（仿真已知精确 = 线圈体积分数 · μ_PtW）。
    X_clean: 同形，干净组织底图（含真牙）。
    metal_mu: 单个体素「纯金属」的 x_M0 幅值。None → 取本样本 x_M0 的峰值（自动匹配
        真实线圈体素，最稳妥）。显式传入时应与数据集 x_M0 的金属尺度一致（CatSim 水
        归一化 μ_PtW=0.72；若 x_M0 用 HU 表示则传 35000）。
    teeth_hu: 判定真牙/致密骨的 HU 阈值（默认 2500）。
    prob: 每个体（样本）的注入概率；<1 时按概率跳过（不做任何改动）。
    frac: 注入的牙齿区域比例（0~1）。<1 时按 (H,W) 空间掩膜随机丢弃部分牙齿列，且该
        掩膜跨 z 一致 → 保住 z 连续性（「部分底图」的随机性，而非逐片抖动）。
    seed: 可选 RNG 种子（同一体多次调用可用同 seed 复现）。

    Returns
    -------
    注毒后的 x_M0（同形同 dtype float32）。真线圈体素不动，牙齿体素被覆盖为 metal 值。
    """
    x_M0 = np.asarray(x_M0, dtype=np.float32)
    X_clean = np.asarray(X_clean, dtype=np.float32)
    rng = np.random.default_rng(seed)

    if prob < 1.0 and float(rng.random()) >= prob:
        return x_M0

    teeth = X_clean > teeth_hu                       # [Z,1,H,W] bool
    if not teeth.any():
        return x_M0

    if frac < 1.0:
        # 空间掩膜 [H,W]，跨 z 广播一致 → 同一牙齿列在整段 z 上同留同弃。
        keep = rng.random(teeth.shape[2:]) < frac
        teeth = teeth & keep[None, None, :, :]

    if not teeth.any():
        return x_M0

    metal_val = float(metal_mu) if metal_mu is not None else float(np.max(x_M0))
    return np.maximum(x_M0, np.where(teeth, metal_val, 0.0).astype(np.float32))


def _self_test() -> bool:
    """本机 numpy 自检：覆盖拼接到牙齿列、真线圈不动、prob=0 原样、z 一致性。"""
    print("[poison self-test]")
    ok = True

    Z, H, W = 3, 8, 8
    rng = np.random.default_rng(0)
    x_M0 = np.zeros((Z, 1, H, W), np.float32)
    x_M0[:, 0, 2:4, 2:4] = 0.72                      # 真线圈体素（颅内中上）
    X_clean = np.zeros((Z, 1, H, W), np.float32) + 100.0
    X_clean[:, 0, 6:8, :] = 3000.0                   # 牙齿：底两行（口腔），跨 z 连续

    out = inject_false_metal(x_M0, X_clean, metal_mu=0.72, prob=1.0, frac=1.0, seed=1)
    t1 = float(out[:, 0, 6:8, :].max()) > 0.0        # 牙齿被覆盖为金属
    t1 &= abs(float(out[:, 0, 2:4, 2:4].max()) - 0.72) < 1e-6  # 真线圈不动（≈0.72）
    ok &= t1
    print(f"  [1] teeth poisoned / coil untouched -> {'OK' if t1 else 'FAIL'}")

    out2 = inject_false_metal(x_M0, X_clean, metal_mu=0.72, prob=0.0, seed=1)
    t2 = np.array_equal(out2, x_M0)                  # prob=0 → 原样
    ok &= t2
    print(f"  [2] prob=0 no-op -> {'OK' if t2 else 'FAIL'}")

    out3 = inject_false_metal(x_M0, X_clean, metal_mu=0.72, prob=1.0, frac=0.5, seed=2)
    # frac<1 的空间掩膜跨 z 一致：任意 (y,x) 在 z 维上要么全注毒要么全不注毒。
    teeth_mask = out3[:, 0, 6:8, :] > 0.0
    t3 = bool((teeth_mask.all(axis=0) | ~teeth_mask.any(axis=0)).all())
    ok &= t3
    print(f"  [3] frac<1 z-consistent -> {'OK' if t3 else 'FAIL'}")

    print(f"  => {'ALL OK' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _self_test() else 1)
