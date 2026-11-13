"""utils —— 归一化（d_ref 冻结）+ SSM 配置。"""

from .normalize import compute_d_ref, normalize_precision
from .config import SSMConfig

__all__ = ["compute_d_ref", "normalize_precision", "SSMConfig"]
