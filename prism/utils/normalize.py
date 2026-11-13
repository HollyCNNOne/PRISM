"""normalize —— 参考数据充分度 d_ref 的冻结计算 + 精度归一化（C5）。

d_ref = median_j d_j：训练集全部体素的 d_j 中位数，训练前一次算好、冻结为标量。
它是**纯数值归一化**（把精度压到单位量级），增益 K 对这个尺度变换不变——d_ref 不是
可学的超参，也不随片/剂量/几何变化。

p = π / d_ref：原始物理精度 π 除以 d_ref 得归一化精度 p。kalman/update.py 里的
观测项 d_j/d_ref 就是这一步的落地（观测精度 π_j = d_j，除以 d_ref 归一化）。

本模块纯 numpy（d_j 场来自 AGIR/仿真，本就是 numpy），本机可自检。
"""

from __future__ import annotations

import numpy as np


def compute_d_ref(d_j_maps) -> float:
    """d_ref = median_j d_j，训练集一次冻结（C5）。

    Parameters
    ----------
    d_j_maps: 可迭代的 d_j 场（numpy 数组，每片一个；形状任意，会 ravel）。

    Returns
    -------
    float 标量（全训练集体素的 d_j 中位数）。
    """
    flat = np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in d_j_maps])
    return float(np.median(flat))


def normalize_precision(pi, d_ref):
    """p = π / d_ref（纯数值归一化，增益不变）。pi 可为 numpy 数组或标量。"""
    return pi / d_ref


def _self_test() -> bool:
    """本机 numpy 自检：d_ref 中位数正确 + 归一化是尺度变换。"""
    print("[normalize self-test]")
    ok = True

    maps = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]), np.array([100.0])]
    d_ref = compute_d_ref(maps)
    t1 = abs(d_ref - 4.0) < 1e-12          # 全 7 个体素：1..6,100 → median = 4（第 4 个）
    ok &= t1
    print(f"  [1] d_ref median (got {d_ref}, want 4.0) -> {'OK' if t1 else 'FAIL'}")

    pi = np.array([1.0, 2.0, 4.0])
    p = normalize_precision(pi, 2.0)
    t2 = np.allclose(p, [0.5, 1.0, 2.0])
    ok &= t2
    print(f"  [2] p = pi/d_ref -> {'OK' if t2 else 'FAIL'}")

    print(f"  => {'ALL OK' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _self_test() else 1)
