"""MNetwork —— 组织流形 M 的深度高斯 SSM 骨架（v9 卡尔曼）。

接口契约：
    forward(S_prev, S_prev_prev, Theta, d_ref, xM0_prev=None, xM0_prevprev=None,
            cold_start=False) → (S_z, K, log_q, delta)

    S_prev / S_prev_prev：上一片 / 上上一片的 KalmanState（p_post 分带 [B, nb, H, W]）；
    Theta = {'x_M0', 'd_j', 'g'} + 可选 'tau_b'/'w_b'（分带观测信任，物理 LFR 派生，冻结）；
    d_ref：训练集冻结的参考数据充分度（C5，调用方传入）；
    返回：新状态 S_z + 卡尔曼增益 K（恒标量场 [B, 1, H, W]，带结构收在精度里）
          + log_q（log 域 Q，供 Q 监督）+ delta（门控 Δ_z，供 Q_target 的 detach）。

    forecast(S_prev, Theta) → M_μ 头的候选预测（独立暴露，供 Q 监督在干净态上评估 M_sg）。

    xM0_prev / xM0_prevprev 是可选参数：骨干 h_z 读 |x_M0_{z-1} − x_M0_{z-2}|，
    需要上一片与上上一片的 x_M0。由 recursion 从 Theta 序列取出并传入；边界片为 None
    → 金属差按 0 处理。

内部四件套 + 解析递归：
    1. 骨干 h_z：读 [|Δμ_{z-1}|, |x_M0_{z-1} − x_M0_{z-2}|]（2ch），浅 2 层 32ch。
    2. 头 Δ_z = g_Δ(h_z) → sigmoid × 0.9（C2：上限 0.9，初始化 ≈ 0.1）。
    3. 头 Q_z = g_Q(h_z, μ_{z-1}) → 输出 log 域（C3）。
    4. 头 M_μ = M_mu(S_{z-1}, Θ_z) → 候选预测（完整状态 + 当前物理量，不只 h_z）。
    5. 预测步 + 更新步 + 增益（调 kalman/）。

分带（论文 §3.5，默认 nb=4，标量 nb=1 退化）：
    M_μ 输入 (nb+5) 通道 [μ; p_post^{b1..bn}; Δμ; x_M0; d_j; g]；
    更新 p_post^b = p_pred^b + τ_b^rad·w_b·d_j/d_ref；
    增益 K = Σ_b p_pred^b / (Σ_b p_pred^b + τ_eff·d_j/d_ref)，τ_eff = Σ_b w_b τ_b^rad。
    缺省 tau_b/w_b → 标量（τ=w=1）。

冷启动（论文 §3.6 Algorithm 1）：cold_start=True 时
    μ_z = M_μ(S_0, Θ_z)（不乘 Δ）；p_post,z^b = d_j/d_ref（各带同标量初始化，非走 τ/w 加权）。

七个约定 C1–C7 落点：
    C1 Δμ 存先验速度（state.py 保证；本模块 delta_mu_z = μ_pred,z − μ_pred,{z-1}，B9 起）。
    C2 Δ_z = sigmoid × 0.9，bias 初始化 −ln8 使 Δ_z ≈ 0.1。
    C3 Q_z 头输出 log 域（forward 里 torch.exp 回正域）。
    C4 p_pred 分母 ε、p_post clip（kalman/predict.py、update.py 保证）。
    C5 d_ref 由调用方传入、冻结，本模块不重算。
    C6 冷启动 S_0 由 recursion/algorithm1.py 用 state.cold_start 造（μ=0, p_post=0[nb], Δμ=0）。
    C7 Q 监督的 detach 属于 training/ 的事，本模块不碰。

预测/更新分离（5.1 #7）：解析预测步 predict_mean / predict_precision 不碰 d_j；
d_j 只在解析更新步 update_precision 进入。M_μ 作为转移算子按 5.1 #6 读完整 Θ_z（含 d_j）。

B9（输出 vs carry 解耦，**只解耦均值、不解耦精度**）：递归只 carry 干净先验均值
（mu_carry=μ_pred）；观测修正 μ_post = K·μ_pred + (1−K)·x̂ 只进「输出」S_z.mu。精度 p_post
保持 B8 不变（后验精度 = p_pred + obs·s，继续进 carry）——它是确定性量、决定 K，不能去掉
观测项（否则 p_pred 卡 0、K 恒 0、完全阴影后验=打穿观测）。修 B8 残余：部分阴影 K→0 时
后验=观测 x̂，若当 carry 传进完全阴影就丢干净先验（bias +77 而非 B6 +5）。见 state.py B9 注释。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..kalman.state import KalmanState
from ..kalman.predict import predict_mean, predict_precision
from ..kalman.update import update_precision, P_MIN, P_MAX
from ..kalman.gain import kalman_gain, shadow_obs_scale


@dataclass
class PredictOut:
    """MNetwork 预测步的产物（不含观测混合，观测由递归驱动注入）。

    递归循环定案（prism-interface-loss-final）：SSM 先预测 μ_pred（carry 先验），
    FIND-Net 以 μ_pred 为输入产出唯一输出 x̂_z，x̂_z 作为观测喂回卡尔曼更新
    μ_post = K·μ_pred + (1−K)·x̂_z。因此预测步与「观测混合」必须分离：混合在
    recursion/algorithm1.py 里由 FIND-Net 观测驱动，不在 MNetwork 内部。

    mu_pred : 预测先验均值 [B, 1, H, W]（FIND-Net 的盲区初始化输入）。
    p_post  : 更新后精度 [B, nb, H, W]（carry 确定性信任量）。
    K       : 卡尔曼增益 [B, 1, H, W]（信先验权重）。
    log_q   : log 域 Q（供 Q 监督）。
    delta   : Δ_z（供 Q_target 的 detach）。
    mu_prev : 上一片 carry 先验均值（供速度 Δμ_z 计算）。
    """
    mu_pred: torch.Tensor
    p_post: torch.Tensor
    K: torch.Tensor
    log_q: torch.Tensor
    delta: torch.Tensor
    mu_prev: torch.Tensor

# C2：Δ_z = sigmoid(·) × Δ_MAX，Δ_MAX = 0.9
DELTA_MAX = 0.9
# C2：初始化让 Δ_z ≈ 0.1 → sigmoid(bias) ≈ 0.1/0.9 = 1/9 → bias = −ln(8)
DELTA_INIT_BIAS = -2.0794
# C3：Q_z 头输出 log 域；bias 初始 log(0.1)（过程噪声初始 Q ≈ 0.1）
LOGQ_INIT_BIAS = -2.3026
# C3：exp 前 clamp log_q，防 g_q 读 HU 尺度 μ 后 log_q 越界 → exp 溢出为 inf，
# 反向 0×inf=NaN（predict_precision 的 q 路径）。q 是过程噪声方差（HU²），
# log_q ∈ [-27.6, ~20] 已覆盖物理范围（Q_target clamp 下限 1e-12 → log −27.6），
# 上下各留余量到 ±30。Q 监督仍在原始 log_q 上做（不 clamp），把饱和值拉回。
LOGQ_MIN = -30.0
LOGQ_MAX = 30.0
# 组织先验 μ 的物理盒（HU）：空气 ≈ −1000，骨/牙 ≈ 2000~3000，Pt-W 金属 35000 属金属区
# （由 x_M0 掩膜 + 下游 FIND-Net 处理，不进 μ）。留少量余量避免在合法骨值边界处截断梯度。
MU_MIN = -1200.0
MU_MAX = 3200.0
# C 消融（H2）C1 门控旧手调参数（AGIR/agir/gate.py 复原）：d_ref_gate = 0.01×d_ref、k=3。
# g(d_j) = 1/(1+(d_j/d_ref_gate)^k) 是先验权重（阴影区→1）。c_head 读 μ 时用 HU_SCALE_GATE 归一化
# （同 r_theta.HU_SCALE=100，把 HU 量级压到 O(1) 与 d_j/d_ref 对齐）。
GATE_D_REF_REL = 0.01
GATE_K = 3.0
HU_SCALE_GATE = 100.0


class MNetwork(nn.Module):
    def __init__(self, nb: int = 4, channels: int = 32, use_velocity: bool = True,
                 gate_reads_dj: bool = False, decouple_mean: bool = False,
                 ablate_xm0: bool = False, ablate_dj: bool = False, ablate_g: bool = False):
        super().__init__()
        self.nb = nb
        # B 消融（H3）B1「去速度」：use_velocity=False 时速度项 Δμ 不进 backbone / M_μ（置 0），
        # 架构与参数量不变，只砍速度信息（位置+速度模型 → 纯位置模型）。
        self.use_velocity = use_velocity
        # 根因修复#2：门控 Δ 读 d_j（gate_reads_dj=True）。Δ 决定 carry vs update，而阴影由
        # d_j/d_ref 定义；旧骨干只读 [速度, 金属几何差]，看不见当前片数据充分度，学不会
        # 「阴影区 Δ→0（carry 上一片干净先验）」。把 d_j/d_ref 加进骨干输入（2ch→3ch）。
        self.gate_reads_dj = gate_reads_dj
        # B9「输出 vs carry 解耦」开关（默认 False = B8 行为：递归 carry 后验 μ_post）。
        # True 时递归只 carry 干净先验 μ_pred（mu_carry），观测 x̂ 只进输出 μ_post。B9 已证
        # 该解耦在训练里开环塌缩（先验无观测锚点），故默认关；保留开关仅为复现
        # 「B8 权重 + 解耦 eval」的混合结果（完全阴影 ~316，逼近 B6 地板），不用于训练。
        self.decouple_mean = decouple_mean
        # A 消融（H1）桥消融：把物理量 x_M0 / d_j / g 从 M_μ（转移算子）输入里去掉（置 0），
        # 只砍「转移算子读物理」这条桥；d_j 在阴影门 s / 卡尔曼增益 K 里的观测模型作用（B8
        # 机制）保留不动——否则 A 消融会把「桥」和「观测模型」混成一个变量，不是干净单变量。
        self.ablate_xm0 = ablate_xm0
        self.ablate_dj = ablate_dj
        self.ablate_g = ablate_g
        # 1. 骨干 h_z：读 [|Δμ_{z-1}|, |x_M0_{z-1} − x_M0_{z-2}|]（+可选 d_j/d_ref），浅 2 层 32ch
        self.backbone = nn.Sequential(
            nn.Conv2d(3 if gate_reads_dj else 2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # 2. 头 Δ_z：1×1 → sigmoid × 0.9（C2）
        self.g_delta = nn.Conv2d(channels, 1, 1)
        # 3. 头 Q_z：读 [h_z, μ_{z-1}]（channels+1），1×1 → log 域（C3）
        self.g_q = nn.Conv2d(channels + 1, 1, 1)
        # 4. 头 M_μ：读 [μ_{z-1}, p_post^{b1..bn}, Δμ_{z-1}, x_M0_z, d_j_z, g_z]（(nb+5)ch）
        self.m_mu = nn.Sequential(
            nn.Conv2d(nb + 5, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 3, padding=1),
        )
        # C 消融（H2）：C1 学习门控的关联强度头 c（读 [μ, d_j/d_ref] → [0,1]）；C0 固定门控的
        # learnable 标量 γ。默认 trust_mode=kalman 不碰它们 → 零梯度 → Adam 不更新（无害）；
        # gated/fixed 时被 trainer 调用、梯度经 R_θ 回流。
        self.c_head = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
        )
        self.fixed_trust = nn.Parameter(torch.tensor(0.5))
        self._init_heads()

    def _init_heads(self):
        # C2：Δ_z 头 bias −ln8 → sigmoid(−ln8)=1/9 → ×0.9 = 0.1；权重小随机避免骨干零梯度
        nn.init.normal_(self.g_delta.weight, std=1e-2)
        nn.init.constant_(self.g_delta.bias, DELTA_INIT_BIAS)
        # C3：Q_z 头 log 域，bias log(0.1)（初始过程噪声 Q ≈ 0.1）
        nn.init.normal_(self.g_q.weight, std=1e-2)
        nn.init.constant_(self.g_q.bias, LOGQ_INIT_BIAS)
        # M_μ 末层小权重，使候选预测初始 ≈ 0（递归初始接近恒等，不伪造组织）
        nn.init.normal_(self.m_mu[-1].weight, std=1e-3)
        nn.init.zeros_(self.m_mu[-1].bias)

    def forecast(self, S_prev, Theta, d_ref=None):
        """M_μ 头：S_{z-1} + Θ_z → 候选预测 μ_candidate（转移算子，5.1 #6 读完整 Θ_z）。

        与 forward 里的 M_μ 完全一致；独立暴露供 Q 监督在干净态上评估 M_sg（detach）。
        d_ref 非 None 时把 d_j 归一化为 d_j/d_ref（与精度更新同尺度）——真实 d_j 是光子
        计数尺度（~1e9），直接进 conv 会主导 M_μ 输入并数值爆炸；除以 d_ref 压到 O(1)。
        """
        d_j = Theta["d_j"]
        if d_ref is not None:
            d_j = d_j / d_ref
        vel = S_prev.delta_mu if self.use_velocity else torch.zeros_like(S_prev.delta_mu)
        # B9：M_μ（转移算子）读干净先验均值 μ_carry（不是观测污染的后验 μ）；精度仍读后验
        # p_post（精度是确定性信任量，保持 B8 不变）。旧状态（融合态/测试构造干净态）无
        # carry 字段时 μ 退回 μ。
        mu_prev = (S_prev.mu_carry
                   if (self.decouple_mean and S_prev.mu_carry is not None)
                   else S_prev.mu)
        p_prev = S_prev.p_post
        # A 消融：物理量桥可选置 0（x_M0/d_j/g 只在 M_μ 这里进转移算子；d_j 的阴影门/增益
        # 作用在 forward 里另算，不受此处影响）。
        x_m0 = Theta["x_M0"] if not self.ablate_xm0 else torch.zeros_like(Theta["x_M0"])
        d_j_in = d_j if not self.ablate_dj else torch.zeros_like(d_j)
        g_in = Theta["g"] if not self.ablate_g else torch.zeros_like(Theta["g"])
        mu_candidate = self.m_mu(torch.cat([
            mu_prev, p_prev, vel,
            x_m0, d_j_in, g_in,
        ], dim=1))
        # 物理盒 clamp：μ 是组织 HU 先验，硬约束到 [MU_MIN, MU_MAX]。
        # 递归 μ_z=(1−Δ)μ_{z-1}+Δ·M_μ 是凸组合（冷启动 μ_1=M_μ），故候选 clamp 后
        # 整条递归的 μ 都在盒内 —— 从结构上杜绝幅值过冲（spectral_norm 只限 Lipschitz，
        # 管不住 M_μ bias 与 backbone/g_Δ 反馈路径把 μ 推到 1e9，进而爆 R_θ/down 损失）。
        return mu_candidate.clamp(min=MU_MIN, max=MU_MAX)

    def compute_gated_trust(self, mu, d_j, d_ref, d_ref_rel=None, k=None, use_c=True):
        """C1 学习门控信任 π = g(d_j)·c（旧 π=g⊙c，AGIR/agir/gate.py 复原）。

        g(d_j) = 1/(1 + (d_j/d_ref_gate)^k)，d_ref_gate = d_ref_rel·d_ref、k=指数 —— 先验权重，
        阴影区→1。c = c_head([μ, d_j/d_ref]) ∈ (0,1) —— 学到的逐体素关联强度。
        mu / d_j: [B, 1, H, W]（μ 为 HU）；d_ref 标量。返回 [B, 1, H, W]。

        d_ref_rel / k / use_c 缺省时用 GATE_D_REF_REL / GATE_K / True（训练/常规评测行为不变）；
        传入时用于「手调旋钮敏感性扫描」—— 冻结 c_head 与 R_θ，只改 g(d_j) 的旋钮，量化
        「调错旋钮的代价 / 关掉学习关联的代价」。
        """
        rel = GATE_D_REF_REL if d_ref_rel is None else d_ref_rel
        kk = GATE_K if k is None else k
        d_ref_gate = rel * d_ref
        r = (d_j / d_ref_gate).clamp(min=0.0)
        rk = r ** kk                            # 指数旋钮
        g = 1.0 / (1.0 + rk)                   # 先验权重
        if not use_c:
            return g
        # μ detach：c_head 是信任头，只把 μ 当「固定参考」读（算关联强度），不回流传给 M_μ。
        # 与 g_q / R_θ 读 μ 时的 detach 同一原则（exp001 懒 μ 三修复）：μ 的结构只该由
        # x̂_final=μ+corr 里那条 ∂/∂μ=1 的干净路径决定，不能经信任头的雅可比 ∂c/∂μ 塑形。
        inp = torch.cat([mu.detach() / HU_SCALE_GATE, d_j / d_ref], dim=1)
        c = torch.sigmoid(self.c_head(inp))
        return g * c

    def predict(self, S_prev, S_prev_prev, Theta, d_ref, xM0_prev=None, xM0_prevprev=None,
                cold_start=False):
        """预测步（推）：从 carry 先验产出 μ_pred / p_post / K / log_q / delta。

        与 forward 的唯一区别：不做观测混合 μ_post = K·μ_pred + (1−K)·x̂。混合留给
        递归驱动（recursion/algorithm1.py），因为观测 x̂ 是 FIND-Net 的输出、依赖 μ_pred
        （递归循环：μ_pred → FIND-Net 输入 → x̂_z → 卡尔曼混合 μ_post）。返回 PredictOut。
        """
        # B9：均值递归 carry 干净先验 μ_pred（mu_carry），不被观测污染；精度递归仍读后验
        # p_post（确定性信任量，保持 B8 不变）。旧状态（融合态/测试构造干净态）无 carry 字段
        # 时 μ 退回 μ。
        mu_prev = (S_prev.mu_carry
                   if (self.decouple_mean and S_prev.mu_carry is not None)
                   else S_prev.mu)
        p_prev = S_prev.p_post

        # 骨干输入（C1：读上一片先验速度；金属几何差）
        delta_mu_prev = S_prev.delta_mu.abs() if self.use_velocity else torch.zeros_like(S_prev.delta_mu)
        if xM0_prev is not None and xM0_prevprev is not None:
            metal_diff = (xM0_prev - xM0_prevprev).abs()
        else:
            metal_diff = torch.zeros_like(delta_mu_prev)
        if self.gate_reads_dj:
            # Δ 读当前片数据充分度 d_j/d_ref（O(1) 归一化，同 M_μ/精度更新的尺度），
            # 让门控能直接「看见阴影」→ 学 Δ→0（carry 上一片干净先验），而非在阴影里信盲的 M_μ。
            h = self.backbone(torch.cat([delta_mu_prev, metal_diff, Theta["d_j"] / d_ref], dim=1))
        else:
            h = self.backbone(torch.cat([delta_mu_prev, metal_diff], dim=1))

        # 头 Δ_z（C2：sigmoid × 0.9）
        delta = torch.sigmoid(self.g_delta(h)) * DELTA_MAX

        # 头 Q_z（C3：log 域 → exp 回正域）。log_q 原样返回给 Q 监督（log 域损失）；
        # 这里只在 exp 前 clamp，防 q=exp(log_q) 溢出 → inf → 反向 0×inf=NaN。
        # h / μ 进 g_q 时 detach：Q 监督只训 g_q 自身权重，梯度不回流传给骨干/μ。
        # 否则 Q 损失对 μ 的梯度（2(log_q−log_qt)·g_q权重，O(1)）比 down 损失对 μ 的梯度
        # （2(x̂_final−X)/HU_SCALE²，~0.09）大 10 倍以上，会主导 μ 的结构并把它推成反相。
        log_q = self.g_q(torch.cat([h.detach(), mu_prev.detach()], dim=1))
        q = torch.exp(log_q.clamp(min=LOGQ_MIN, max=LOGQ_MAX))

        # 头 M_μ（完整 carry 先验 S_{z-1} + Θ_z，输入 (nb+5) 通道）
        mu_candidate = self.forecast(S_prev, Theta, d_ref)

        # 预测步（推）—— 从 carry 先验推，解析部分不碰 d_j
        p_pred = predict_precision(p_prev, delta, q)

        # 分带观测信任（物理 LFR 派生，冻结输入；缺省 → 标量 τ=w=1）
        tau_b = Theta.get("tau_b")
        w_b = Theta.get("w_b")
        tau_eff = None
        if tau_b is not None and w_b is not None:
            tau_eff = (w_b * tau_b).sum(dim=1, keepdim=True)

        # B8 阴影感知观测精度：金属伪影是确定性 bias（不是光子噪声），观测在完全阴影里
        # 不可信，但 d_j/d_ref 只反映光子数、不反映「这条线穿过金属」。给观测精度乘尖锐门
        # s(r)：r=d_j/d_ref < r0(1/9) 完全阴影 → s→0 → K→1 信先验（别把打穿的观测漏进 μ）；
        # r>r0 部分阴影 → s→1 → K→0 信观测。s 同时进 gain 与 update（打穿的观测既不进均值
        # 也不进精度）。修 B7 完全阴影 μ_bias +686 的根因：K 太软（p_sum 小），漏观测。
        s = shadow_obs_scale(Theta["d_j"], d_ref)

        # 增益（定义式，恒标量 [B, 1, H, W]）
        K = kalman_gain(p_pred, Theta["d_j"], d_ref, tau_eff, s=s)

        if cold_start:
            # 冷启动（§3.6）：先验 μ_pred = M_μ(S_0, Θ_z)（不乘 Δ）；p_post,z^b = d_j/d_ref 各带同标量。
            # p_pred = 0（S_0 精度为 0）→ K = 0（无先验，观测充分决定首片）。
            mu_pred = mu_candidate
            p_post_z = (Theta["d_j"] / d_ref).expand(-1, self.nb, -1, -1).clamp(
                min=P_MIN, max=P_MAX
            )
        else:
            # 预测均值 + 更新精度（补）—— d_j 只在这里进入
            mu_pred = predict_mean(mu_prev, delta, mu_candidate)
            p_post_z = update_precision(p_pred, Theta["d_j"], d_ref, tau_b=tau_b, w_b=w_b, s=s)

        return PredictOut(mu_pred=mu_pred, p_post=p_post_z, K=K, log_q=log_q,
                          delta=delta, mu_prev=mu_prev)

    def forward(self, S_prev, S_prev_prev, Theta, d_ref, xM0_prev=None, xM0_prevprev=None,
                cold_start=False, x_hat=None):
        """预测 + 观测混合（后向兼容薄包装）。

        递归循环定案后，观测混合移到 recursion/algorithm1.py（观测 = FIND-Net 输出，
        依赖 μ_pred）。本方法仅为单测/旧调用保留：predict() 后直接混合传入的 x_hat。
        """
        pr = self.predict(S_prev, S_prev_prev, Theta, d_ref, xM0_prev=xM0_prev,
                          xM0_prevprev=xM0_prevprev, cold_start=cold_start)

        # 观测修正（卡尔曼补）——只进「输出」不进「carry」：
        #   μ_post = K·μ_pred + (1−K)·x̂
        # K 是「信先验」权重：阴影 d_j→0 ⇒ K→1 ⇒ 信 μ_pred；充分 d_j 大 ⇒ K→0 ⇒ 信 x̂。
        # 冷启动 p_pred=0 ⇒ K=0 ⇒ μ_1=x̂_1。x_hat=None 时保持纯预测（后向兼容：输出=先验）。
        mu_post = (pr.K * pr.mu_pred + (1.0 - pr.K) * x_hat) if x_hat is not None else pr.mu_pred

        # C1：速度 Δμ_z。B8（decouple_mean=False，状态 carry 后验）用后验差
        #   μ_post,z − μ_post,{z-1}；B9（decouple_mean=True，状态 carry 先验）用先验差
        #   μ_pred,z − μ_pred,{z-1}（不观测污染）。
        delta_mu_z = (pr.mu_pred - pr.mu_prev) if self.decouple_mean else (mu_post - pr.mu_prev)

        S_z = KalmanState(mu=mu_post, p_post=pr.p_post, delta_mu=delta_mu_z,
                          mu_carry=(pr.mu_pred if self.decouple_mean else None))
        return S_z, pr.K, pr.log_q, pr.delta
