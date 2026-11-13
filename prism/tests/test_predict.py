"""test_predict —— 预测步：均值混合、Q↑→p_pred↓、Δ=Q=0→精度无损继承（分带逐带传播）。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from prism.kalman.predict import predict_mean, predict_precision


def test_predict_mean_blend():
    torch.manual_seed(0)
    mu_prev = torch.rand(2, 1, 8, 8)
    cand = torch.rand(2, 1, 8, 8)
    delta0 = torch.zeros_like(mu_prev)
    assert torch.allclose(predict_mean(mu_prev, delta0, cand), mu_prev)  # Δ=0 → μ_z = μ_{z-1}
    delta1 = torch.ones_like(mu_prev)
    assert torch.allclose(predict_mean(mu_prev, delta1, cand), cand)      # Δ=1 → μ_z = M_μ


def test_predict_precision_q_up_pdown():
    p = torch.full((1, 4, 4, 4), 2.0)   # [B, nb, H, W]
    d = torch.full((1, 1, 4, 4), 0.5)   # [B, 1, H, W] 跨带广播
    q_small = predict_precision(p, d, torch.tensor(0.1))
    q_big = predict_precision(p, d, torch.tensor(10.0))
    assert torch.all(q_big < q_small)  # Q 增大 → p_pred 衰减


def test_predict_precision_inherit_when_d0_q0():
    torch.manual_seed(0)
    p = torch.rand(1, 4, 4, 4) + 1.0
    d = torch.zeros(1, 1, 4, 4)
    q = torch.zeros(1, 1, 4, 4)
    pp = predict_precision(p, d, q)
    assert torch.allclose(pp, p, atol=1e-5)  # Δ=0,Q=0 → 精度无损继承


def test_predict_precision_scalar_degenerate():
    p = torch.full((1, 1, 4, 4), 2.0)
    d = torch.full((1, 1, 4, 4), 0.5)
    pp = predict_precision(p, d, torch.tensor(0.1))
    assert pp.shape == (1, 1, 4, 4)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
