"""training —— Q 监督损失 + 下游 R_θ 去除损失 + 三轴训练循环（Phase 4）。"""

from .losses import (
    Q_WINDOW,
    Q_Z_WINDOW,
    compute_clean_residual,
    residual_to_q_target,
    q_supervision_loss,
    downstream_reconstruction_loss,
    physics_consistency_loss,
)
from .trainer import SSMTrainer

__all__ = [
    "Q_WINDOW",
    "Q_Z_WINDOW",
    "compute_clean_residual",
    "residual_to_q_target",
    "q_supervision_loss",
    "downstream_reconstruction_loss",
    "physics_consistency_loss",
    "SSMTrainer",
]
