"""frequency —— 分带观测精度的物理量（论文 §3.5，Phase 5）。

原始径向 LFR → w_b（信息份额）/ τ_b^rad（每带信任）/ τ_eff（有效信任）。
分带常量复制自 AGIR 的 agir/lir_freq.py；本模块纯 numpy 自包含（本机可自检）。
"""

from .banding import (
    RADIAL_EDGES,
    N_ANGULAR,
    D_REF_FREQ,
    K_FREQ,
    phi_freq,
    radial_lfr_from_params,
    radial_lfr_to_weights,
    from_agir_lfr,
)

__all__ = [
    "RADIAL_EDGES",
    "N_ANGULAR",
    "D_REF_FREQ",
    "K_FREQ",
    "phi_freq",
    "radial_lfr_from_params",
    "radial_lfr_to_weights",
    "from_agir_lfr",
]
