"""test_frequency —— frequency 包（radial_lfr_from_params + 分带常量）的 numpy 自检（本机可跑）。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from frequency.banding import radial_lfr_from_params, RADIAL_EDGES, N_ANGULAR


def test_radial_lfr_shape_and_range():
    theta = np.full((8, 8), np.deg2rad(30.0))
    smaj = np.full((8, 8), 0.10)
    smin = np.full((8, 8), 0.03)
    lfr = radial_lfr_from_params(theta, smaj, smin)
    assert lfr.shape == (4, 8, 8)
    assert float(lfr.min()) >= 0.0 and float(lfr.max()) <= 1.0


def test_radial_edges_four_bands():
    assert len(RADIAL_EDGES) == 5  # 5 条边 → 4 带
    assert N_ANGULAR == 8


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
