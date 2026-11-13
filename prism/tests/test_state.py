"""test_state —— KalmanState 形状 / dtype / 冷启动（分带 nb=4 默认，nb=1 退化）。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from prism.kalman.state import KalmanState


def test_shape_and_dtype_banded():
    B, H, W = 2, 8, 8
    s = KalmanState.cold_start((B, H, W), nb=4)
    assert s.mu.shape == (B, 1, H, W)
    assert s.p_post.shape == (B, 4, H, W)
    assert s.delta_mu.shape == (B, 1, H, W)
    # B9：carry 字段（冷启动 S_0 无先验 → 也为 0）
    assert s.mu_carry.shape == (B, 1, H, W)
    for t in (s.mu, s.p_post, s.delta_mu, s.mu_carry):
        assert t.dtype == torch.float32
        assert torch.all(t == 0.0)


def test_shape_scalar_degenerate():
    B, H, W = 2, 8, 8
    s = KalmanState.cold_start((B, H, W), nb=1)
    for t in (s.mu, s.p_post, s.delta_mu, s.mu_carry):
        assert t.shape == (B, 1, H, W), t.shape


def test_carry_fields_default_none():
    # 显式构造（融合态 / 测试干净态）无 carry 字段 → None（向后兼容）
    s = KalmanState(mu=torch.zeros(1, 1, 4, 4), p_post=torch.zeros(1, 4, 4, 4),
                    delta_mu=None)
    assert s.mu_carry is None


def test_three_fields_only():
    s = KalmanState.cold_start((1, 4, 4))
    assert hasattr(s, "mu") and hasattr(s, "p_post") and hasattr(s, "delta_mu")
    assert hasattr(s, "mu_carry")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
