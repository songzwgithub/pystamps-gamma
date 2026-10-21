from __future__ import annotations

import numpy as np

from pystamps.pipeline.stage8_sbas_krig import (
    _empirical_variogram_from_pairs,
    _fit_exp_variogram,
    _official_time_design,
    _ols_projector,
    _ordinary_kriging_batch_full50,
    _ordinary_kriging_one,
    _ordinary_kriging_operator_1d,
)


def test_stamps_edge_deformation_model_is_removed_exactly():
    day = 730486.0 + np.arange(0, 49 * 12, 12, dtype=float)
    years, A = _official_time_design(day)
    assert A.shape == (49, 5)
    assert years.shape == (49,)

    rng = np.random.default_rng(4)
    coeff = rng.normal(size=(128, 5))
    dph = coeff @ A.T
    P = _ols_projector(A)
    estimated = dph @ P.T
    residual = dph - estimated @ A.T
    assert np.max(np.abs(residual)) < 1e-9


def test_temporal_ordinary_kriging_operator_has_unbiased_weights():
    years = np.linspace(24.0, 25.6, 49)
    W = _ordinary_kriging_operator_1d(
        years,
        a=0.8,
        nugget=0.2,
        sill=1.5,
        nmax=27,
        exclude_self=False,
    )
    assert W.shape == (49, 49)
    assert np.max(np.abs(W.sum(axis=1) - 1.0)) < 1e-8

    Wloo = _ordinary_kriging_operator_1d(
        years,
        a=0.8,
        nugget=0.2,
        sill=1.5,
        nmax=27,
        exclude_self=True,
    )
    assert np.max(np.abs(Wloo.sum(axis=1) - 1.0)) < 1e-8
    assert np.max(np.abs(np.diag(Wloo))) < 1e-12


def test_vectorized_spatial_ordinary_kriging_matches_scalar():
    rng = np.random.default_rng(5)
    B, k = 8, 12
    neigh = rng.normal(size=(B, k, 2)) * 1000.0
    target = rng.normal(size=(B, 2)) * 100.0
    z = rng.normal(size=(B, k))

    fast = _ordinary_kriging_batch_full50(
        neigh,
        z,
        target,
        a=15000.0,
        nugget=0.3,
        sill=1.2,
    )
    slow = np.asarray([
        _ordinary_kriging_one(
            neigh[i], z[i], target[i], 15000.0, 0.3, 1.2
        )
        for i in range(B)
    ])
    np.testing.assert_allclose(fast, slow, rtol=1e-10, atol=1e-10)


def test_exp_variogram_fit_recovers_clean_model():
    h = np.linspace(0.02, 1.4, 30)
    a0 = 0.55
    c10 = 1.7
    c20 = 0.2
    g = c10 * (1.0 - np.exp(-h / a0)) + c20
    a, c1, c2 = _fit_exp_variogram(h, g, 0.25, 1.0, 2.0)
    assert abs(a - a0) < 1e-3
    assert abs(c1 - c10) < 1e-3
    assert abs(c2 - c20) < 1e-3


def test_variogram_binning_is_finite():
    rng = np.random.default_rng(6)
    d = rng.uniform(0, 1.5, 10000)
    g = rng.gamma(2.0, 0.4, 10000)
    h, v = _empirical_variogram_from_pairs(d, g, 1.5, 25)
    assert h.size > 5
    assert h.shape == v.shape
    assert np.all(np.isfinite(h))
    assert np.all(np.isfinite(v))
