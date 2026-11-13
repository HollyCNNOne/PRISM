"""data —— Theta 序列 Dataset / collate + 假阳性注入（Phase 4）。"""

from .dataset import ThetaDataset, collate_theta, make_theta_sequence_loader
from .poison import inject_false_metal, TEETH_HU_DEFAULT

__all__ = ["ThetaDataset", "collate_theta", "make_theta_sequence_loader",
           "inject_false_metal", "TEETH_HU_DEFAULT"]
