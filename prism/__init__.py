"""PRISM —— 物理条件化深度 MAR 域适应（核心库）。

组织流形 M 的深度高斯 SSM / 卡尔曼递归骨架（v9 定案，论文 §3.5 分带默认）。PyTorch 实现。
分十块：models（M 网络 / FIND-Net 包装 / 翻译器 / 三个头）、kalman（状态 / 预测 / 更新 /
增益 / 融合）、recursion（Algorithm 1 前向后向）、training（Q 监督 + 三轴）、inference、
frequency（分带卡尔曼）、utils、data、tests、scripts。

SSM 主体已补全（Phase 1–5）：kalman/ 精度分带 nb=4（nb=1 标量退化）、models/m_network.py
M_μ 输入 (nb+5) 通道、frequency/banding.py 分带物理量、utils（d_ref 冻结 + 配置）、
data（Theta 序列 Dataset）、training（Q 监督 + 三轴训练）、inference（融合 μ/K）。
"""

from __future__ import annotations

from .kalman.state import KalmanState
from .models.m_network import MNetwork

__all__ = ["KalmanState", "MNetwork"]

__version__ = "0.1.0"
