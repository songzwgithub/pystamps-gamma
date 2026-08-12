from __future__ import annotations

import numpy as np
cimport cython
cimport numpy as cnp
from cython.parallel cimport prange
from libc.math cimport atan2, ceil, cos, sin, sqrt, isfinite


cdef double _PI = 3.14159265358979323846264338327950288
cdef double _QUARTER_PI = _PI / 4.0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _ps_topofit_row(
    cnp.complex128_t[:, :] cpx_view,
    cnp.float64_t[:, :] bp_view,
    Py_ssize_t row,
    Py_ssize_t n_col,
    int trial_n,
    cnp.float64_t[:] k_view,
    cnp.float64_t[:] c_view,
    cnp.float64_t[:] coh_view,
    cnp.complex128_t[:, :] res_view,
) noexcept nogil:
    cdef Py_ssize_t col
    cdef Py_ssize_t trial_ix
    cdef int best_trial
    cdef double bperp_min
    cdef double bperp_max
    cdef double bperp_range
    cdef double denom
    cdef double denom2
    cdef double den_lin
    cdef double weight
    cdef double wb
    cdef double coh
    cdef double best_coh
    cdef double K
    cdef double C
    cdef double coh0
    cdef double phase
    cdef double cs
    cdef double sn
    cdef double ph_re
    cdef double ph_im
    cdef double sum_re
    cdef double sum_im
    cdef double offset_re
    cdef double offset_im
    cdef double mean_re
    cdef double mean_im
    cdef double res_re
    cdef double res_im
    cdef double angle_re
    cdef double angle_im
    cdef double mopt_num
    cdef double mopt

    bperp_min = bp_view[row, 0]
    bperp_max = bp_view[row, 0]
    denom = 0.0
    for col in range(n_col):
        if bp_view[row, col] < bperp_min:
            bperp_min = bp_view[row, col]
        if bp_view[row, col] > bperp_max:
            bperp_max = bp_view[row, col]
        ph_re = cpx_view[row, col].real
        ph_im = cpx_view[row, col].imag
        denom += sqrt(ph_re * ph_re + ph_im * ph_im)

    if denom == 0.0:
        denom = 1.0
    bperp_range = bperp_max - bperp_min
    if bperp_range == 0.0:
        bperp_range = 1.0

    best_trial = -trial_n
    best_coh = -1.0
    for trial_ix in range(2 * trial_n + 1):
        sum_re = 0.0
        sum_im = 0.0
        for col in range(n_col):
            phase = (bp_view[row, col] / bperp_range) * _QUARTER_PI * (trial_ix - trial_n)
            cs = cos(phase)
            sn = sin(phase)
            ph_re = cpx_view[row, col].real
            ph_im = cpx_view[row, col].imag
            sum_re += (ph_re * cs) + (ph_im * sn)
            sum_im += (ph_im * cs) - (ph_re * sn)
        coh = sqrt(sum_re * sum_re + sum_im * sum_im) / denom
        if coh > best_coh:
            best_coh = coh
            best_trial = trial_ix - trial_n

    K = (_QUARTER_PI / bperp_range) * best_trial

    offset_re = 0.0
    offset_im = 0.0
    for col in range(n_col):
        phase = K * bp_view[row, col]
        cs = cos(phase)
        sn = sin(phase)
        ph_re = cpx_view[row, col].real
        ph_im = cpx_view[row, col].imag
        res_re = (ph_re * cs) + (ph_im * sn)
        res_im = (ph_im * cs) - (ph_re * sn)
        offset_re += res_re
        offset_im += res_im

    den_lin = 0.0
    mopt_num = 0.0
    for col in range(n_col):
        phase = K * bp_view[row, col]
        cs = cos(phase)
        sn = sin(phase)
        ph_re = cpx_view[row, col].real
        ph_im = cpx_view[row, col].imag
        res_re = (ph_re * cs) + (ph_im * sn)
        res_im = (ph_im * cs) - (ph_re * sn)
        angle_re = (res_re * offset_re) + (res_im * offset_im)
        angle_im = (res_im * offset_re) - (res_re * offset_im)
        weight = sqrt(ph_re * ph_re + ph_im * ph_im)
        wb = weight * bp_view[row, col]
        den_lin += wb * wb
        mopt_num += wb * (weight * atan2(angle_im, angle_re))

    if den_lin == 0.0:
        den_lin = 1.0
    mopt = mopt_num / den_lin
    K += mopt

    mean_re = 0.0
    mean_im = 0.0
    denom2 = 0.0
    for col in range(n_col):
        phase = K * bp_view[row, col]
        cs = cos(phase)
        sn = sin(phase)
        ph_re = cpx_view[row, col].real
        ph_im = cpx_view[row, col].imag
        res_re = (ph_re * cs) + (ph_im * sn)
        res_im = (ph_im * cs) - (ph_re * sn)
        mean_re += res_re
        mean_im += res_im
        denom2 += sqrt(res_re * res_re + res_im * res_im)
        res_view[row, col] = res_re + (1j * res_im)

    if denom2 == 0.0:
        denom2 = 1.0
    C = atan2(mean_im, mean_re)
    coh0 = sqrt(mean_re * mean_re + mean_im * mean_im) / denom2

    k_view[row] = K
    c_view[row] = C
    coh_view[row] = coh0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _ps_topofit_row_invariant_core(
    cnp.complex128_t[:, :] cpx_view,
    cnp.float64_t[:] bp_vec,
    cnp.float64_t[:, :] basis_cs,
    cnp.float64_t[:, :] basis_sn,
    Py_ssize_t row,
    Py_ssize_t n_col,
    int trial_n,
    double bperp_range,
    cnp.complex128_t[:, :] res_view,
    bint store_phase,
    double* out_k,
    double* out_c,
    double* out_coh,
) noexcept nogil:
    cdef Py_ssize_t col
    cdef Py_ssize_t trial_ix
    cdef int best_trial
    cdef double denom
    cdef double denom2
    cdef double den_lin
    cdef double weight
    cdef double wb
    cdef double coh
    cdef double best_coh
    cdef double K
    cdef double C
    cdef double coh0
    cdef double phase
    cdef double cs
    cdef double sn
    cdef double ph_re
    cdef double ph_im
    cdef double sum_re
    cdef double sum_im
    cdef double offset_re
    cdef double offset_im
    cdef double mean_re
    cdef double mean_im
    cdef double res_re
    cdef double res_im
    cdef double angle_re
    cdef double angle_im
    cdef double mopt_num
    cdef double mopt

    denom = 0.0
    for col in range(n_col):
        ph_re = cpx_view[row, col].real
        ph_im = cpx_view[row, col].imag
        denom += sqrt(ph_re * ph_re + ph_im * ph_im)

    if denom == 0.0:
        denom = 1.0

    best_trial = -trial_n
    best_coh = -1.0
    for trial_ix in range(2 * trial_n + 1):
        sum_re = 0.0
        sum_im = 0.0
        for col in range(n_col):
            cs = basis_cs[trial_ix, col]
            sn = basis_sn[trial_ix, col]
            ph_re = cpx_view[row, col].real
            ph_im = cpx_view[row, col].imag
            sum_re += (ph_re * cs) + (ph_im * sn)
            sum_im += (ph_im * cs) - (ph_re * sn)
        coh = sqrt(sum_re * sum_re + sum_im * sum_im) / denom
        if coh > best_coh:
            best_coh = coh
            best_trial = trial_ix - trial_n

    K = (_QUARTER_PI / bperp_range) * best_trial

    offset_re = 0.0
    offset_im = 0.0
    for col in range(n_col):
        phase = K * bp_vec[col]
        cs = cos(phase)
        sn = sin(phase)
        ph_re = cpx_view[row, col].real
        ph_im = cpx_view[row, col].imag
        res_re = (ph_re * cs) + (ph_im * sn)
        res_im = (ph_im * cs) - (ph_re * sn)
        offset_re += res_re
        offset_im += res_im

    den_lin = 0.0
    mopt_num = 0.0
    for col in range(n_col):
        phase = K * bp_vec[col]
        cs = cos(phase)
        sn = sin(phase)
        ph_re = cpx_view[row, col].real
        ph_im = cpx_view[row, col].imag
        res_re = (ph_re * cs) + (ph_im * sn)
        res_im = (ph_im * cs) - (ph_re * sn)
        angle_re = (res_re * offset_re) + (res_im * offset_im)
        angle_im = (res_im * offset_re) - (res_re * offset_im)
        weight = sqrt(ph_re * ph_re + ph_im * ph_im)
        wb = weight * bp_vec[col]
        den_lin += wb * wb
        mopt_num += wb * (weight * atan2(angle_im, angle_re))

    if den_lin == 0.0:
        den_lin = 1.0
    mopt = mopt_num / den_lin
    K += mopt

    mean_re = 0.0
    mean_im = 0.0
    denom2 = 0.0
    for col in range(n_col):
        phase = K * bp_vec[col]
        cs = cos(phase)
        sn = sin(phase)
        ph_re = cpx_view[row, col].real
        ph_im = cpx_view[row, col].imag
        res_re = (ph_re * cs) + (ph_im * sn)
        res_im = (ph_im * cs) - (ph_re * sn)
        mean_re += res_re
        mean_im += res_im
        denom2 += sqrt(res_re * res_re + res_im * res_im)
        if store_phase:
            res_view[row, col] = res_re + (1j * res_im)

    if denom2 == 0.0:
        denom2 = 1.0
    C = atan2(mean_im, mean_re)
    coh0 = sqrt(mean_re * mean_re + mean_im * mean_im) / denom2

    if out_k != NULL:
        out_k[0] = K
    if out_c != NULL:
        out_c[0] = C
    if out_coh != NULL:
        out_coh[0] = coh0


@cython.boundscheck(False)
@cython.wraparound(False)
def accumulate_weighted_grid(
    cnp.ndarray[cnp.complex64_t, ndim=2] ph_weight,
    cnp.ndarray[cnp.int64_t, ndim=1] grid_lin,
    int n_i,
    int n_j,
    int threads=1,
):
    cdef Py_ssize_t n_ps = ph_weight.shape[0]
    cdef Py_ssize_t n_ifg = ph_weight.shape[1]
    cdef Py_ssize_t row
    cdef Py_ssize_t col
    cdef long long idx
    cdef long long grid_size = <long long>(n_i * n_j)
    cdef int thread_count = threads if threads > 0 else 1
    cdef cnp.ndarray[cnp.complex64_t, ndim=2] out_flat = np.zeros((n_i * n_j, n_ifg), dtype=np.complex64)
    cdef cnp.complex64_t[:, :] ph_view = ph_weight
    cdef cnp.int64_t[:] grid_view = grid_lin
    cdef cnp.complex64_t[:, :] out_view = out_flat

    for col in prange(n_ifg, nogil=True, schedule="static", num_threads=thread_count):
        for row in range(n_ps):
            idx = grid_view[row]
            if 0 <= idx < grid_size:
                out_view[idx, col] += ph_view[row, col]

    return out_flat.reshape((n_i, n_j, n_ifg))


@cython.boundscheck(False)
@cython.wraparound(False)
def ps_topofit_batch_generic(
    cnp.ndarray[cnp.complex128_t, ndim=2] cpxphase,
    cnp.ndarray[cnp.float64_t, ndim=2] bperp,
    double n_trial_wraps,
    int threads=1,
):
    from pystamps.pipeline import ported

    # Keep the native module entry point, but delegate full-output topofit to
    # the exact Python selector semantics used by the Stage 2 parity path.
    return ported._ps_topofit_batch_generic(
        np.asarray(cpxphase, dtype=np.complex128),
        np.asarray(bperp, dtype=np.float64),
        float(n_trial_wraps),
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def ps_topofit_batch_row_invariant(
    cnp.ndarray[cnp.complex128_t, ndim=2] cpxphase,
    cnp.ndarray[cnp.float64_t, ndim=1] bperp_vec,
    double n_trial_wraps,
    int threads=1,
):
    from pystamps.pipeline import ported

    return ported._ps_topofit_batch_row_invariant(
        np.asarray(cpxphase, dtype=np.complex128),
        np.tile(np.asarray(bperp_vec, dtype=np.float64), (np.asarray(cpxphase).shape[0], 1)),
        float(n_trial_wraps),
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def ps_topofit_coh_row_invariant(
    cnp.ndarray[cnp.complex128_t, ndim=2] cpxphase,
    cnp.ndarray[cnp.float64_t, ndim=1] bperp_vec,
    double n_trial_wraps,
    int threads=1,
):
    from pystamps.pipeline import ported

    return ported._ps_topofit_batch_row_invariant(
        np.asarray(cpxphase, dtype=np.complex128),
        np.tile(np.asarray(bperp_vec, dtype=np.float64), (np.asarray(cpxphase).shape[0], 1)),
        float(n_trial_wraps),
    )[2]


@cython.boundscheck(False)
@cython.wraparound(False)
def histogram_with_centers(
    cnp.ndarray[cnp.float64_t, ndim=1] values,
    cnp.ndarray[cnp.float64_t, ndim=1] centers,
):
    cdef Py_ssize_t n_value = values.shape[0]
    cdef Py_ssize_t n_center = centers.shape[0]
    cdef Py_ssize_t ix
    cdef Py_ssize_t lo
    cdef Py_ssize_t hi
    cdef Py_ssize_t mid
    cdef double value
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.zeros((n_center,), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] mids
    cdef cnp.float64_t[:] val_view = values
    cdef cnp.float64_t[:] ctr_view = centers
    cdef cnp.float64_t[:] out_view = out
    cdef cnp.float64_t[:] mid_view

    if n_center == 0:
        return out
    if n_center == 1:
        out_view[0] = float(n_value)
        return out

    mids = np.empty((n_center - 1,), dtype=np.float64)
    mid_view = mids
    for ix in range(n_center - 1):
        mid_view[ix] = (ctr_view[ix] + ctr_view[ix + 1]) / 2.0

    for ix in range(n_value):
        value = val_view[ix]
        if not isfinite(value):
            continue
        lo = 0
        hi = n_center - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if mid_view[mid] < value:
                lo = mid + 1
            else:
                hi = mid
        out_view[lo] += 1.0

    return out
