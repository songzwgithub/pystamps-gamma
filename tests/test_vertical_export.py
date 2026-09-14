from __future__ import annotations

import numpy as np

from pystamps.vertical_export import _vertical_factor


def test_away_positive_los_to_vertical_sign():
    inc = np.deg2rad(np.array([30.0, 40.0]))
    expected = 1.0 / np.cos(inc)

    assert np.allclose(
        _vertical_factor(inc, "up"),
        -expected,
    )
    assert np.allclose(
        _vertical_factor(inc, "down"),
        expected,
    )
