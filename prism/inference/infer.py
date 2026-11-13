"""inference —— 推理入口（Phase 3）。

复用 recursion.forward_backward，输出融合后的 (μ_z, K_z)。§3.7 的增益加权拉拽
η₃ K ⊙ (x̂ − μ) 由 FIND-Net 侧消费，本模块只提供 μ/K 接口，不实现 R_θ。
"""

from __future__ import annotations

from ..recursion.algorithm1 import forward_backward


def infer(m_net, thetas, d_ref):
    """SSM 推理：前向 + 后向 + 精度加权融合。

    thetas: [Z] dict（物理量，冻结）；d_ref: 训练集冻结的参考数据充分度。
    返回 dict：
        mu_z: [Z] 个 [B, 1, H, W]（融合态均值，切片顺序对齐）；
        K_fwd / K_bwd: [Z] 个 [B, 1, H, W] 增益（前向/后向趟，§3.7 由 FIND-Net 选方向）。
    """
    (fwd, bwd, fused, fwd_gains, bwd_gains, *_tails) = forward_backward(m_net, thetas, d_ref)
    return {"mu_z": [s.mu for s in fused], "K_fwd": fwd_gains, "K_bwd": bwd_gains}
