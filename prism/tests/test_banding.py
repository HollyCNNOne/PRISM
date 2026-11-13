"""test_banding —— 分带观测精度的 numpy 自检（本机可跑，无 torch）。

覆盖：φ_k^freq 范围/单调、w_b 归一化和为 1、τ_eff∈[0,1]、nb=1 退化、平坦退化。
导入方式：把 PRISM/ 目录加进 sys.path 后 import frequency.banding（绕过 prism/__init__
的 torch 依赖，本机无 torch 也能跑）。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from frequency.banding import phi_freq, radial_lfr_to_weights, D_REF_FREQ


def test_phi_range_and_monotone():
    t = np.linspace(0.0, 1.0, 1001)
    phi = phi_freq(t)
    assert float(phi.min()) >= 0.0 and float(phi.max()) <= 1.0
    assert bool(np.all(np.diff(phi) >= 0.0))
    assert abs(float(phi_freq(D_REF_FREQ)) - 0.5) < 1e-9


def test_w_b_sums_to_one_and_tau_eff_range():
    lfr = np.array([0.9, 0.6, 0.3, 0.1])[:, None, None]
    w_b, tau_b, tau_eff = radial_lfr_to_weights(lfr)
    assert abs(float(w_b.sum()) - 1.0) < 1e-9
    assert float(tau_eff.min()) >= 0.0 and float(tau_eff.max()) <= 1.0


def test_nb1_degenerate():
    lfr = np.array([0.7])[:, None, None]
    w_b, tau_b, tau_eff = radial_lfr_to_weights(lfr)
    assert abs(float(w_b[0, 0, 0]) - 1.0) < 1e-9
    assert abs(float(tau_eff[0, 0, 0]) - float(tau_b[0, 0, 0])) < 1e-9


def test_flat_response():
    lfr = np.full((4, 3, 3), 0.8)
    w_b, tau_b, tau_eff = radial_lfr_to_weights(lfr)
    assert np.allclose(w_b, 0.25)
    assert np.allclose(tau_eff, phi_freq(0.8))


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
