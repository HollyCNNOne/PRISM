"""test_gain —— 增益：d_j→0→K→1，d_j 大→K→0，K∈[0,1]（Σ_b 求和 + τ_eff）。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from prism.kalman.gain import kalman_gain


def test_gain_shadow_to_one():
    p_pred = torch.full((1, 4, 4, 4), 1.0)   # Σ_b = 4
    d_j = torch.zeros(1, 1, 4, 4)
    K = kalman_gain(p_pred, d_j, 1.0)
    assert K.shape == (1, 1, 4, 4)
    assert torch.allclose(K, torch.ones_like(K), atol=1e-5)  # 阴影 d_j→0 → K→1


def test_gain_sufficient_to_zero():
    p_pred = torch.full((1, 4, 4, 4), 1.0)
    d_j = torch.full((1, 1, 4, 4), 1e6)
    K = kalman_gain(p_pred, d_j, 1.0)
    assert torch.all(K < 1e-4)  # 充分区 d_j 大 → K→0


def test_gain_banded_tau_eff():
    p_pred = torch.full((1, 4, 4, 4), 1.0)   # Σ_b = 4
    d_j = torch.full((1, 1, 4, 4), 4.0)
    tau_eff = torch.full((1, 1, 4, 4), 0.5)
    K = kalman_gain(p_pred, d_j, 2.0, tau_eff=tau_eff)
    assert torch.allclose(K, torch.full((1, 1, 4, 4), 0.8))  # 4/(4+0.5*(4/2))=4/5


def test_gain_scalar_degenerate():
    p_pred = torch.full((1, 1, 4, 4), 1.0)
    d_j = torch.full((1, 1, 4, 4), 4.0)
    K = kalman_gain(p_pred, d_j, 2.0)
    assert torch.allclose(K, torch.full_like(K, 1.0 / 3.0))


def test_gain_in_unit_interval():
    torch.manual_seed(0)
    p_pred = torch.rand(4, 4, 8, 8) + 0.1
    d_j = torch.rand(4, 1, 8, 8) * 100
    K = kalman_gain(p_pred, d_j, 5.0)
    assert K.shape == (4, 1, 8, 8)
    assert torch.all(K >= 0) and torch.all(K <= 1)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
