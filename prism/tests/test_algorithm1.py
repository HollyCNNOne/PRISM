"""test_algorithm1 —— 端到端递归：前向+后向+融合，形状、融合精度、无 NaN（9 元组返回）。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from prism.models.m_network import MNetwork
from prism.recursion.algorithm1 import forward_backward


def _thetas(Z, B, H, W, banded=False):
    thetas = []
    for _ in range(Z):
        th = {
            "x_M0": torch.rand(B, 1, H, W),
            "d_j": torch.rand(B, 1, H, W) * 10,
            "g": torch.rand(B, 1, H, W),
        }
        if banded:
            th["tau_b"] = torch.rand(B, 4, H, W)
            th["w_b"] = torch.rand(B, 4, H, W)
            th["w_b"] = th["w_b"] / th["w_b"].sum(dim=1, keepdim=True).clamp(min=1e-12)
        thetas.append(th)
    return thetas


def test_forward_backward_shapes():
    torch.manual_seed(0)
    Z, B, H, W = 5, 2, 8, 8
    m = MNetwork()
    fwd, bwd, fused, fg, bg, flq, blq, fd, bd = forward_backward(
        m, _thetas(Z, B, H, W, banded=True), d_ref=5.0
    )
    assert len(fwd) == len(bwd) == len(fused) == Z
    assert len(fg) == len(bg) == len(flq) == len(blq) == len(fd) == len(bd) == Z
    for s in fused:
        assert s.mu.shape == (B, 1, H, W) and s.p_post.shape == (B, 4, H, W)
        assert s.delta_mu is None  # 速度不融合
    for K in fg:
        assert K.shape == (B, 1, H, W)


def test_forward_backward_scalar_degenerate():
    torch.manual_seed(0)
    Z, B, H, W = 3, 1, 6, 6
    m = MNetwork(nb=1)
    fwd, bwd, fused, fg, bg, *_ = forward_backward(m, _thetas(Z, B, H, W), d_ref=5.0)
    for s in fused:
        assert s.p_post.shape == (B, 1, H, W)


def test_fused_precision_exceeds_each():
    torch.manual_seed(0)
    m = MNetwork()
    fwd, bwd, fused, _, _, _, _, _, _ = forward_backward(m, _thetas(3, 1, 6, 6), d_ref=5.0)
    for f, b, fu in zip(fwd, bwd, fused):
        assert torch.all(fu.p_post >= f.p_post - 1e-6)
        assert torch.all(fu.p_post >= b.p_post - 1e-6)


def test_no_nan_end_to_end():
    torch.manual_seed(0)
    m = MNetwork()
    fwd, bwd, fused, fg, bg, _, _, _, _ = forward_backward(
        m, _thetas(4, 2, 8, 8, banded=True), d_ref=5.0
    )
    for s in fused:
        assert torch.isfinite(s.mu).all() and torch.isfinite(s.p_post).all()
    for K in fg + bg:
        assert torch.isfinite(K).all() and torch.all(K >= 0) and torch.all(K <= 1)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
