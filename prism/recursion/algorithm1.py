"""Algorithm 1 —— 沿 z 轴的完整卡尔曼递归（前向趟 + 后向趟 + 精度加权融合）。

递归循环定案（prism-interface-loss-final）：观测不再是冻结的脏图，而是 FIND-Net 的输出
x̂_z —— 它依赖 SSM 的预测先验 μ_pred，因此观测必须在循环**内**逐片产生：

    μ_pred_z = MNet.predict(S_{z-1}, Θ_z)          # SSM 预测（carry 先验）
    x̂_z      = FIND-Net(CT_ma=脏图, LIct=μ_pred_z, Mask)   # FIND-Net 唯一输出
    μ_post_z  = K·μ_pred_z + (1−K)·x̂_z               # 卡尔曼混合（观测修正）
    S_z       = (μ_post_z, p_post_z, Δμ_z)          # carry 到 z+1

observer(mu_pred, z) -> x̂ 是调用方注入的 FIND-Net 前向（本模块不 import FIND-Net，
保持递归与下游网络解耦）。observer 为 None 且 x_hats 为 None 时退化为纯 SSM（开环，
供 z 连续性消融/诊断脚本）。

前向趟：z = 1..Z，冷启动 S_0（C6），首片 cold_start=True（§3.6）。
后向趟：z = Z..1，同一递归，Θ 序列倒序，末片 cold_start=True。
融合：fuse(S_z^fwd, S_z^bwd) = 精度加权平均均值、精度相加；速度不融合。

返回 RecursionResult（dataclass，支持 9 元组解包以向后兼容旧脚本）：
    前 9 个字段顺序与旧版 9 元组一致（fwd/bwd/fused + 两方向 gains/logq/delta），
    新增 fwd_x_hats / bwd_x_hats（FIND-Net 输出观测，[Z] 列表，无 observer 时 None）。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..kalman.state import KalmanState
from ..kalman.fuse import fuse
from ..models.m_network import MNetwork, MU_MIN, MU_MAX


@dataclass
class RecursionResult:
    fwd: list
    bwd: list
    fused: list
    fwd_gains: list
    bwd_gains: list
    fwd_logq: list
    bwd_logq: list
    fwd_delta: list
    bwd_delta: list
    fwd_x_hats: list | None = None   # FIND-Net 输出（前向趟观测）[Z]；无 observer 时 None
    bwd_x_hats: list | None = None   # FIND-Net 输出（后向趟观测）[Z]；无 observer 时 None

    def __iter__(self):
        # 向后兼容旧 9 元组解包：fwd, bwd, fused, fg, bg, flq, blq, fd, bd = res
        yield self.fwd
        yield self.bwd
        yield self.fused
        yield self.fwd_gains
        yield self.bwd_gains
        yield self.fwd_logq
        yield self.bwd_logq
        yield self.fwd_delta
        yield self.bwd_delta


def _effective_gain(m_net, pr, theta, d_ref, gain_mode):
    """A3 消融：卡尔曼混合的「信先验」权重 K 的可替换来源。

    "kalman"（现状）：pr.K —— 物理涌现的卡尔曼增益（p_pred/(p_pred + τ_eff·d_j/d_ref·s)）。
    "gated"（经验门控）：g(d_j) = 1/(1 + (d_j/(0.01·d_ref))^3)，阴影 d_j→0 ⇒ K→1（信先验），
        充分 d_j 大 ⇒ K→0（信观测）。复用 m_network.compute_gated_trust(use_c=False) 的手调门。
    "fixed"（常数）：可学习标量 m_net.fixed_trust 广播——空间不自适应，best-case 黑盒。

    只替换「均值混合权重 K」；卡尔曼精度 p_pred/p_post 与 Q/Δ 头全部保持不变（单变量消融）。
    """
    if gain_mode == "kalman":
        return pr.K
    if gain_mode == "gated":
        return m_net.compute_gated_trust(None, theta["d_j"], d_ref, use_c=False)
    if gain_mode == "fixed":
        return m_net.fixed_trust.expand_as(pr.K)
    raise ValueError(f"未知 gain_mode: {gain_mode!r}")


def _run_pass(m_net, thetas, d_ref, step, per_slice=False, observer=None, x_hats=None,
              cold_start_mu=None, gain_mode="kalman"):
    """单趟递归。step = −1 前向（prev = z−1），step = +1 后向（prev = z+1）。

    per_slice=True：无递归基线（B 消融）——每片都从 S0 冷启动，不携带跨片状态、
    不读邻居 x_M0（金属几何差置 0），彻底「逐片独立」。M_μ 仍读当前片自身的 Θ_z
    （x_M0/d_j/g），只砍跨片状态演化。

    observer：callable(mu_pred, z) -> x̂_z。非 None 时在每片预测后调 FIND-Net 产出观测
    x̂_z，再卡尔曼混合 μ_post = K·μ_pred + (1−K)·x̂_z（递归循环定案）。
    x_hats：旧接口的冻结观测切片 [Z] 个 [B,1,H,W]（observer 为 None 时用它直接当观测，
    向后兼容旧脚本的「脏图直接当观测」用法）。

    返回 (states, gains, log_qs, deltas, x_hats_out) 列表，与原始切片顺序对齐；
    x_hats_out 为每片实际用到的观测（observer 产出或 x_hats 直取，纯 SSM 时 None）。
    """
    Z = len(thetas)
    order = list(range(Z)) if step < 0 else list(range(Z - 1, -1, -1))
    B, _, H, W = thetas[0]["d_j"].shape
    S0 = KalmanState.cold_start(
        (B, H, W), device=thetas[0]["d_j"].device, dtype=thetas[0]["d_j"].dtype,
        nb=m_net.nb,
    )
    # Stage 2 冷启动暖启动：S0.μ 不用全零，用首片观测暖启动（强拉短序列收敛）。
    # cold_start_mu 已 [B,1,H,W]；clamp 到组织物理盒，防脏图金属 HU 喂进 M_μ 首层。
    if cold_start_mu is not None:
        S0.mu = cold_start_mu.clamp(min=MU_MIN, max=MU_MAX)
    prev2 = prev1 = S0
    states = [None] * Z
    gains = [None] * Z
    log_qs = [None] * Z
    deltas = [None] * Z
    x_hats_out = [None] * Z
    for z in order:
        theta = thetas[z]
        if per_slice:
            pr = m_net.predict(S0, S0, theta, d_ref, xM0_prev=None, xM0_prevprev=None,
                               cold_start=True)
        else:
            i1 = z + step
            i2 = z + 2 * step
            xM0_prev = thetas[i1]["x_M0"] if 0 <= i1 < Z else None
            xM0_prevprev = thetas[i2]["x_M0"] if 0 <= i2 < Z else None
            pr = m_net.predict(prev1, prev2, theta, d_ref,
                               xM0_prev=xM0_prev, xM0_prevprev=xM0_prevprev,
                               cold_start=(z == order[0]))

        # 观测：优先 observer（FIND-Net 输出），否则退回旧 x_hats 直取，否则纯 SSM。
        if observer is not None:
            x_hat = observer(pr.mu_pred, z)
        elif x_hats is not None:
            x_hat = x_hats[z]
        else:
            x_hat = None

        # 卡尔曼混合（观测修正，只进「输出」不进「carry」，B9 语义见 m_network.predict）
        # A3：混合权重 K 可替换（kalman/gated/fixed），卡尔曼精度 p_post 保持原样。
        K = _effective_gain(m_net, pr, theta, d_ref, gain_mode)
        mu_post = (K * pr.mu_pred + (1.0 - K) * x_hat) if x_hat is not None else pr.mu_pred
        delta_mu_z = (pr.mu_pred - pr.mu_prev) if m_net.decouple_mean else (mu_post - pr.mu_prev)
        S_z = KalmanState(mu=mu_post, p_post=pr.p_post, delta_mu=delta_mu_z,
                          mu_carry=(pr.mu_pred if m_net.decouple_mean else None))

        states[z] = S_z
        gains[z] = K
        log_qs[z] = pr.log_q
        deltas[z] = pr.delta
        x_hats_out[z] = x_hat
        prev2 = prev1
        prev1 = S_z
    return states, gains, log_qs, deltas, x_hats_out


def forward_backward(m_net, thetas, d_ref, recursion="bidirectional", observer=None,
                     x_hats=None, cold_start_mus=None, gain_mode="kalman"):
    """前向 + 后向 + 精度加权融合（默认双向）；B 消融支持单向 / 无递归。

    recursion:
        "bidirectional"（默认）：前向 + 后向 + 精度加权融合（论文主形式）。
        "forward"：只前向趟，融合态 = 前向态（无后向平滑，B 消融「单向」）。
        "per_slice"：无递归基线——每片独立冷启动（B 消融「无递归」）。

    observer：callable(mu_pred, z) -> x̂_z（FIND-Net 观测，递归循环定案）；None 时
    退回 x_hats 直取或纯 SSM（开环）。返回 RecursionResult（含 fwd_x_hats/bwd_x_hats）。
    """
    if recursion == "per_slice":
        s, g, lq, dl, xh = _run_pass(m_net, thetas, d_ref, step=-1, per_slice=True,
                                     observer=observer, x_hats=x_hats, gain_mode=gain_mode)
        return RecursionResult(s, s, s, g, g, lq, lq, dl, dl, fwd_x_hats=xh, bwd_x_hats=xh)
    if recursion == "forward":
        cmu = cold_start_mus[0] if cold_start_mus else None
        fwd, fg, flq, fdl, fxh = _run_pass(m_net, thetas, d_ref, step=-1,
                                           observer=observer, x_hats=x_hats,
                                           cold_start_mu=cmu, gain_mode=gain_mode)
        return RecursionResult(fwd, fwd, fwd, fg, fg, flq, flq, fdl, fdl,
                               fwd_x_hats=fxh, bwd_x_hats=fxh)
    cmu_f = cold_start_mus[0] if cold_start_mus else None
    cmu_b = cold_start_mus[-1] if cold_start_mus else None
    fwd, fwd_gains, fwd_logq, fwd_delta, fwd_xh = _run_pass(
        m_net, thetas, d_ref, step=-1, observer=observer, x_hats=x_hats, cold_start_mu=cmu_f,
        gain_mode=gain_mode)
    bwd, bwd_gains, bwd_logq, bwd_delta, bwd_xh = _run_pass(
        m_net, thetas, d_ref, step=+1, observer=observer, x_hats=x_hats, cold_start_mu=cmu_b,
        gain_mode=gain_mode)
    fused = [fuse(f, b) for f, b in zip(fwd, bwd)]
    return RecursionResult(fwd, bwd, fused, fwd_gains, bwd_gains,
                           fwd_logq, bwd_logq, fwd_delta, bwd_delta,
                           fwd_x_hats=fwd_xh, bwd_x_hats=bwd_xh)
