"""test_m_network —— M 网络：前向形状、Δ_z 头、Q_z 头、K 范围、无 NaN、梯度反传、冷启动。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from prism.kalman.state import KalmanState
from prism.models.m_network import MNetwork, DELTA_MAX


def _theta(B, H, W, banded=False):
    th = {
        "x_M0": torch.rand(B, 1, H, W),
        "d_j": torch.rand(B, 1, H, W) * 10,
        "g": torch.rand(B, 1, H, W),
    }
    if banded:
        th["tau_b"] = torch.rand(B, 4, H, W)
        th["w_b"] = torch.rand(B, 4, H, W)
        th["w_b"] = th["w_b"] / th["w_b"].sum(dim=1, keepdim=True).clamp(min=1e-12)
    return th


def test_forward_shapes_and_k_range():
    torch.manual_seed(0)
    B, H, W = 2, 8, 8
    m = MNetwork()  # nb=4
    S0 = KalmanState.cold_start((B, H, W))
    S, K, log_q, delta = m(S0, S0, _theta(B, H, W, banded=True), d_ref=5.0)
    assert S.mu.shape == (B, 1, H, W)
    assert S.p_post.shape == (B, 4, H, W)
    assert S.delta_mu.shape == (B, 1, H, W)
    assert K.shape == (B, 1, H, W)
    assert log_q.shape == (B, 1, H, W)
    assert delta.shape == (B, 1, H, W)
    assert torch.all(K >= 0) and torch.all(K <= 1)


def test_forward_scalar_degenerate():
    torch.manual_seed(0)
    B, H, W = 2, 8, 8
    m = MNetwork(nb=1)
    S0 = KalmanState.cold_start((B, H, W), nb=1)
    S, K, log_q, delta = m(S0, S0, _theta(B, H, W), d_ref=5.0)
    assert S.p_post.shape == (B, 1, H, W)
    assert K.shape == (B, 1, H, W)


def test_cold_start_mu_equals_forecast():
    torch.manual_seed(0)
    B, H, W = 2, 8, 8
    m = MNetwork()
    S0 = KalmanState.cold_start((B, H, W))
    theta = _theta(B, H, W)
    S, K, log_q, delta = m(S0, S0, theta, d_ref=5.0, cold_start=True)
    mu_c = m.forecast(S0, theta, d_ref=5.0)
    assert torch.allclose(S.mu, mu_c)  # 冷启动 μ_1 = M_μ(S_0, Θ_1)，不乘 Δ


def test_cold_start_uniform_bands():
    torch.manual_seed(0)
    B, H, W = 2, 8, 8
    m = MNetwork()
    S0 = KalmanState.cold_start((B, H, W))
    theta = _theta(B, H, W, banded=True)
    d_ref = 5.0
    S, K, log_q, delta = m(S0, S0, theta, d_ref=d_ref, cold_start=True)
    expected = (theta["d_j"] / d_ref).expand(-1, 4, -1, -1).clamp(min=1e-6, max=1e6)
    assert torch.allclose(S.p_post, expected)  # 冷启动 p_post,1^b = d_j/d_ref 各带同标量


def test_delta_head_range_and_init():
    torch.manual_seed(0)
    m = MNetwork()
    h = m.backbone(torch.zeros(1, 2, 4, 4))
    delta = torch.sigmoid(m.g_delta(h)) * DELTA_MAX
    assert torch.all(delta >= 0) and torch.all(delta <= DELTA_MAX)
    assert abs(delta.mean().item() - 0.1) < 0.02  # C2：初始 ≈ 0.1


def test_q_head_positive_logdomain():
    torch.manual_seed(0)
    m = MNetwork()
    h = torch.zeros(1, 32, 4, 4)
    mu = torch.zeros(1, 1, 4, 4)
    log_q = m.g_q(torch.cat([h, mu], dim=1))
    q = torch.exp(log_q)
    assert torch.all(q > 0)
    assert torch.all(torch.isfinite(log_q))


def test_no_nan():
    torch.manual_seed(0)
    m = MNetwork()
    S0 = KalmanState.cold_start((2, 8, 8))
    S, K, log_q, delta = m(S0, S0, _theta(2, 8, 8), d_ref=5.0)
    for t in (S.mu, S.p_post, S.delta_mu, K, log_q, delta):
        assert torch.isfinite(t).all()


def test_gradients_flow():
    torch.manual_seed(0)
    m = MNetwork()
    S0 = KalmanState.cold_start((2, 8, 8))
    S, K, log_q, delta = m(S0, S0, _theta(2, 8, 8), d_ref=5.0)
    loss = S.mu.pow(2).sum() + K.pow(2).sum() + delta.sum()
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(torch.any(g != 0) for g in grads)


def test_b9_carry_invariant_to_observation():
    """B9：观测 x̂ 只改变「输出」后验 μ，不改变「carry」先验均值 μ_carry；精度 p_post 保持
    后验（确定性信任量，含观测项），对 x̂ 不变（它不读 x̂，只读 d_j/Q/上一片 p_post）。"""
    torch.manual_seed(0)
    B, H, W = 2, 8, 8
    m = MNetwork(decouple_mean=True)
    S0 = KalmanState.cold_start((B, H, W))
    theta = _theta(B, H, W)
    x1 = torch.randn(B, 1, H, W)
    x2 = torch.randn(B, 1, H, W)
    S1, K1, _, _ = m(S0, S0, theta, d_ref=5.0, x_hat=x1)
    S2, K2, _, _ = m(S0, S0, theta, d_ref=5.0, x_hat=x2)
    # carry（先验均值）不随观测变
    assert torch.allclose(S1.mu_carry, S2.mu_carry)
    # 精度是后验、不含 x̂ → 对 x̂ 不变（回归 B9 初版 bug：误把精度也解耦成 p_carry 卡 0）
    assert torch.allclose(S1.p_post, S2.p_post)
    assert torch.any(S1.p_post > 0)
    # 输出（后验）随观测变
    assert not torch.allclose(S1.mu, S2.mu)
    # 输出 = 后验 = K·μ_pred + (1−K)·x̂
    assert torch.allclose(S1.mu, K1 * S1.mu_carry + (1.0 - K1) * x1)
    assert torch.allclose(S2.mu, K2 * S2.mu_carry + (1.0 - K2) * x2)
    # 先验速度 = μ_pred − μ_prev（S0 冷启动 μ_prev=0）
    assert torch.allclose(S1.delta_mu, S1.mu_carry - S0.mu_carry)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
