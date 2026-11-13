"""test_fuse —— 融合：精度逐带相加、均值用总精度（带和）加权、一方精度 0 → 退化为另一方。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from prism.kalman.state import KalmanState
from prism.kalman.fuse import fuse


def _state(mu, p):
    return KalmanState(mu=mu, p_post=p, delta_mu=None)


def test_fuse_precision_sums():
    p1 = torch.full((1, 4, 4, 4), 2.0)
    p2 = torch.full((1, 4, 4, 4), 4.0)
    f = fuse(_state(torch.ones(1, 1, 4, 4), p1), _state(torch.full((1, 1, 4, 4), 3.0), p2))
    assert torch.allclose(f.p_post, p1 + p2)  # 逐带相加


def test_fuse_mean_total_precision_weighted():
    mu1 = torch.zeros(1, 1, 4, 4)
    p1 = torch.full((1, 4, 4, 4), 1.0)   # Σ_b = 4
    mu2 = torch.full((1, 1, 4, 4), 10.0)
    p2 = torch.full((1, 4, 4, 4), 2.0)   # Σ_b = 8
    f = fuse(_state(mu1, p1), _state(mu2, p2))
    assert torch.allclose(f.mu, torch.full((1, 1, 4, 4), 80.0 / 12.0))  # (4·0+8·10)/12


def test_fuse_scalar_degenerate():
    mu1 = torch.zeros(1, 1, 4, 4)
    p1 = torch.full((1, 1, 4, 4), 1.0)
    mu2 = torch.full((1, 1, 4, 4), 10.0)
    p2 = torch.full((1, 1, 4, 4), 3.0)
    f = fuse(_state(mu1, p1), _state(mu2, p2))
    assert torch.allclose(f.mu, (p1 * mu1 + p2 * mu2) / (p1 + p2))


def test_fuse_degenerate_one_zero():
    torch.manual_seed(0)
    mu1 = torch.rand(1, 1, 4, 4)
    p1 = torch.rand(1, 4, 4, 4) + 1.0
    mu2 = torch.rand(1, 1, 4, 4)
    p2 = torch.zeros(1, 4, 4, 4)
    f = fuse(_state(mu1, p1), _state(mu2, p2))
    assert torch.allclose(f.mu, mu1, atol=1e-5)  # p2=0 → 退化为 mu1
    assert torch.allclose(f.p_post, p1)
    assert f.delta_mu is None  # 速度不融合


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
