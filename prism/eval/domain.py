"""domain.py — 评分域的唯一真源（**所有指标必须调用本函数，禁止手写**）。

铁律
----
    评分域 = 非金属 ∩ 非回填 ∩ 非空气

为什么必须排除回填区
------------------
`trainer._backfill` 把金属核外扩 r 像素的那一圈**直接替换成 Baseline**：

    Final = (1 − dilate(mask>0.5, r)) · Pred + dilate(...) · Baseline

那一圈既不是网络输出（内容来自 Baseline），又不归网络负责（梯度恒为 0）。
若评分时只排除金属核（mask>ε）而保留回填环，就等于**用 Baseline 的错去罚网络**：
实测 8px 环可以把 large 的 FIND-Net MAE 从 56.8 抬到 97.0、p99 从 305 抬到 2939。

历史教训：这个错在 2026-11 前后重复出现近 10 次，每次都是因为**各脚本手写评分域**。
故收敛成此函数。
"""
from __future__ import annotations

import numpy as np

try:  # scipy 在 agir / pzneuro 都有
    from scipy.ndimage import maximum_filter
except Exception:  # pragma: no cover
    maximum_filter = None


def tissue_mask(mask, gt, r: int = 8, metal_eps: float = 1e-3, air_hu: float = -500.0):
    """返回布尔评分域，形状与输入一致（2D 或 3D 均可）。

    Parameters
    ----------
    mask : np.ndarray
        金属体积分数场（0~1），2D ``[H,W]`` 或 3D ``[Z,H,W]``。
    gt : np.ndarray
        干净真值（HU），形状同 ``mask``。
    r : int
        回填外扩半径（必须与 ``SSMConfig.backfill_r`` 一致，默认 8）。
    metal_eps : float
        ``mask > metal_eps`` 视为金属。
    air_hu : float
        ``gt <= air_hu`` 视为空气/背景，不参与评分。

    Returns
    -------
    np.ndarray[bool]  同形状；True = 可评分体素（非金属 ∩ 非回填 ∩ 非空气）。
    """
    mask = np.asarray(mask)
    gt = np.asarray(gt)
    if mask.shape != gt.shape:
        raise ValueError(f"tissue_mask: shape mismatch {mask.shape} vs {gt.shape}")

    metal = mask > metal_eps
    if r > 0:
        if mask.ndim == 2:
            size = (2 * r + 1, 2 * r + 1)
        elif mask.ndim == 3:
            size = (1, 2 * r + 1, 2 * r + 1)      # 只沿 (H,W) 膨胀，不跨 z
        else:
            raise ValueError(f"tissue_mask: 仅支持 2D/3D，得到 ndim={mask.ndim}")
        # 与 _backfill 的 max_pool2d(k=2r+1, pad=r) 等价（square structuring element）
        if maximum_filter is None:  # 退化：无 scipy 时至少排除金属
            backfill = metal.copy()
        else:
            backfill = maximum_filter(metal, size=size, mode="constant", cval=False)
    else:
        backfill = metal

    return (~backfill) & (gt > air_hu)
