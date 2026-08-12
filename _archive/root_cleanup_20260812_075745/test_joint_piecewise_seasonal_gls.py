#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

main_path = Path(__file__).with_name('calc_joint_piecewise_seasonal_gls.py')
spec = importlib.util.spec_from_file_location('joint_model', main_path)
if spec is None or spec.loader is None:
    raise RuntimeError('Cannot load main module')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

dates=[]
d=datetime(2020,1,5)
while d<=datetime(2024,12,25):
    dates.append(d)
    d += timedelta(days=12)

design=module.build_design(dates, seasonal_harmonics=1, seasonal_period_days=365.2425)
X=design.matrix
n,p=X.shape
assert p == 1 + len(design.model_years) + 2
assert np.linalg.matrix_rank(X) == p

rho=0.70
idx=np.arange(n)
C=0.012**2 * rho ** np.abs(idx[:,None]-idx[None,:])
eigval,eigvec=np.linalg.eigh(C)
W=(eigvec/np.sqrt(np.maximum(eigval,1e-12))[None,:]).T
context=module.WhitenedContext(
    mask=np.ones(n,bool), X=X, Xw=W@X, whitener=W,
    covariance_meta={}, rank=np.linalg.matrix_rank(X),
)

rng=np.random.default_rng(20260730)
B=400
beta_true=np.zeros((B,p))
beta_true[:,0]=rng.normal(0,0.2,B)
slopes=rng.normal(0,0.05,(B,len(design.model_years)))
beta_true[:,design.year_column_indices]=slopes
beta_true[:,design.annual_sin_index]=rng.normal(0.12,0.02,B)
beta_true[:,design.annual_cos_index]=rng.normal(-0.04,0.02,B)
L=np.linalg.cholesky(C+np.eye(n)*1e-12)
y=beta_true@X.T + rng.normal(size=(B,n))@L.T

# Sparse large outliers; the estimator should remain finite and reasonably accurate.
for _ in range(400):
    r=int(rng.integers(B)); c=int(rng.integers(n))
    y[r,c] += rng.normal(0,1.0)

result=module.robust_joint_gls_batch(
    y, context, robust=True, huber_c=1.345, weight_floor=0.05,
    max_iterations=8, convergence=1e-6,
)
est=result['beta'][:,design.year_column_indices]
median_error=float(np.nanmedian(np.abs(est-slopes)))
print('epochs, parameters:', n, p)
print('fit valid:', int(np.count_nonzero(result['fit_valid'])), '/', B)
print('median annual-slope absolute error:', median_error, 'rad/yr')
print('median downweighted modes:', float(np.median(result['downweighted_mode_count'])))
assert np.count_nonzero(result['fit_valid']) == B
assert median_error < 0.03
assert np.count_nonzero(result['downweighted_mode_count']) > 0
print('Synthetic joint piecewise-seasonal GLS test passed')
