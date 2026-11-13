"""test_update —— 更新步：p_post = p_pred + 观测项（分带 τ_b^rad·w_b·d_j/d_ref），clip。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from prism.kalman.update import update_precision, P_MIN, P_MAX


def test_update_adds_observation_flat():
    p_pred = torch.full((1, 4, 4, 4), 1.0)
    d_j = torch.full((1, 1, 4, 4), 4.0)
    p_post = update_precision(p_pred, d_j, 2.0)  # 无 tau/w → 观测项广播到各带
    assert torch.allclose(p_post, p_pred + d_j / 2.0)


def test_update_banded_weighted():
    p_pred = torch.full((1, 4, 4, 4), 1.0)
    d_j = torch.full((1, 1, 4, 4), 4.0)
    tau_b = torch.full((1, 4, 4, 4), 0.5)
    w_b = torch.full((1, 4, 4, 4), 0.25)
    p_post = update_precision(p_pred, d_j, 2.0, tau_b=tau_b, w_b=w_b)
    assert torch.allclose(p_post, p_pred + 0.25)  # 0.5*0.25*(4/2)=0.25


def test_update_scalar_degenerate():
    p_pred = torch.full((1, 1, 4, 4), 1.0)
    d_j = torch.full((1, 1, 4, 4), 4.0)
    p_post = update_precision(p_pred, d_j, 2.0)
    assert torch.allclose(p_post, p_pred + 2.0)


def test_update_clip_high():
    p_pred = torch.full((1, 4, 4, 4), 1e9)
    p_post = update_precision(p_pred, torch.zeros(1, 1, 4, 4), 1.0)
    assert torch.all(p_post <= P_MAX)


def test_update_clip_low():
    p_post = update_precision(torch.zeros(1, 4, 4, 4), torch.zeros(1, 1, 4, 4), 1.0)
    assert torch.all(p_post >= P_MIN)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
