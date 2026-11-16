"""config —— SSM 配置 dataclass（最小化，日志并入训练器不在此处）。

单一事实源：DELTA_MAX / DELTA_INIT_BIAS / LOGQ_INIT_BIAS 从 models/m_network.py 导入，
避免魔法常量两处漂移。d_ref 不是配置项——由 utils/normalize.py 从训练集算好、冻结。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models.m_network import DELTA_MAX, DELTA_INIT_BIAS, LOGQ_INIT_BIAS


@dataclass
class SSMConfig:
    """组织流形 M 的深度高斯 SSM 配置（论文 §3.5 分带默认）。

    nb: 分带数（4=分带默认，1=标量退化，对应消融 A）。
    channels: 骨干 h_z / M_μ 头的隐藏宽度。
    delta_max / delta_init_bias / logq_init_bias: Δ_z / Q_z 头的界限与初始化（C2/C3）。
    lambda_q: Q 监督损失权重（training/losses.py 用）。
    lambda_pc: 物理一致性惩罚 L_pc 权重（training/losses.physics_consistency_loss）。
    lambda_down: 下游 FIND-Net 去除损失 L_down 权重（training/losses.downstream_reconstruction_loss）。
    lambda_mu: 组织先验 μ 的直接保真损失 L_mu 权重（默认 0=关；>0 逼 μ≈X_clean，
        训出「干净引导」供 FIND-Net 盲区参考——递归循环定案后保留、监督本片）。
    trust_mode: C 消融（H2）R_θ 信任信号机制——"kalman"（后验总精度 p_post_sum，现状）/
        "gated"（旧 π=g⊙c，gate.py 复原）/"fixed"（learnable 标量 γ）。默认 kalman。
    recursion: B 消融（H3）递归模式——"bidirectional"（双向，默认）/ "forward"（单向）/
        "per_slice"（无递归基线）。
    pc_metal_hu_thr: L_pc 判定「金属 HU」的阈值（真牙/骨 <2500，Pt-W 远高于此）。
    pc_d_thr_frac: L_pc 的 d_thr = pc_d_thr_frac · d_ref（默认 0.5）。
    fixed_prior_p: Q_target 干净态 M_sg 的固定先验精度（归一化，= 物理 d_ref；Plan B）。
    d_ref: 冻结的参考数据充分度（None 表示尚未冻结，训练前由 compute_d_ref 填充）。
    """

    nb: int = 4
    channels: int = 32
    delta_max: float = DELTA_MAX
    delta_init_bias: float = DELTA_INIT_BIAS
    logq_init_bias: float = LOGQ_INIT_BIAS
    lambda_q: float = 1.0
    lambda_pc: float = 0.1
    lambda_down: float = 1.0
    lambda_mu: float = 0.0
    trust_mode: str = "kalman"
    recursion: str = "bidirectional"
    pc_metal_hu_thr: float = 2500.0
    pc_d_thr_frac: float = 0.5
    fixed_prior_p: float = 1.0
    d_ref: float | None = None
    cold_start_init: str = "zero"   # 冷启动 S0.μ 初始化："zero"（全零，现状）|"obs"（Baseline 脏图）|"xm0"（金属先验）
    # ---- 消融矩阵新增开关（Stage 2 用，默认=基线） ----
    gain_mode: str = "kalman"       # A3 增益替换："kalman"（物理卡尔曼增益，现状）/
                                    #   "gated"（经验门控 g(d_j)=1/(1+(d_j/(0.01·d_ref))^3)）/
                                    #   "fixed"（可学习常数 γ，best-case 黑盒）
    findnet_lict_source: str = "mu"  # A2 去先验：FIND-Net 的 LIct 输入——"mu"（SSM 先验 μ_pred，现状）/
                                    #   "li"（经典 LI-MAR，SSM 算了 μ 但不喂 FIND-Net）
    binarize_mask: bool = False      # B1 硬切掩膜：金属掩膜二值化 0/1（默认连续 vf）
    mask_bin_thr: float = 0.5        # B1 二值化阈值（vf > thr → 金属）
    loss_include_metal: bool = False  # B1 金属内算 Loss：金属区也监督（默认 (1−mask) 弃疗金属）
    lambda_ssim: float = 0.1         # v1.3：L_down 内 SSIM 高频结构约束权重（0=关；baseline 定案 0.1）
    backfill_r: int = 8              # v1.3：最终输出金属回填的形态学膨胀半径 px（0=关）
