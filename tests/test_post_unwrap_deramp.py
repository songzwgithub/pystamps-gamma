from __future__ import annotations

import numpy as np

from pystamps.pipeline.post_unwrap_deramp import (
    _robust_plane,
    _robust_through_origin,
)


def test_robust_plane_recovers_spatial_gradient():
    rng = np.random.default_rng(42)
    x = rng.uniform(-2.0, 2.0, 1000)
    y = rng.uniform(-1.5, 1.5, 1000)
    z = (
        0.7
        + 0.35 * x
        - 0.18 * y
        + rng.normal(0.0, 0.01, x.size)
    )
    z[:8] += 20.0

    beta, r2, n = _robust_plane(
        x,
        y,
        z,
        max_iter=8,
        clip=3.5,
    )

    assert n > 950
    assert r2 > 0.99
    assert np.allclose(
        beta,
        [0.7, 0.35, -0.18],
        atol=0.01,
    )


def test_robust_through_origin_recovers_rate():
    rng = np.random.default_rng(7)
    t = np.linspace(-4.0, 4.0, 80)
    y = 0.42 * t + rng.normal(0.0, 0.01, t.size)
    y[[4, 50]] += np.array([2.0, -3.0])

    slope, keep = _robust_through_origin(
        t,
        y,
        max_iter=8,
        clip=3.5,
    )

    assert np.count_nonzero(keep) >= 75
    assert abs(slope - 0.42) < 0.01
