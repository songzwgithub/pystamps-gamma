#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V6 — MintPy-style network inversion benchmark on the existing 4:1 IFG stack.

NO spatial filtering.
NO IFG deletion.
NO resampling/downsampling.
NO environment/package installation.

Scientific objective
--------------------
Test whether a mature MintPy-style interferogram-network inversion improves the
current result BEFORE moving further upstream to SARvey/Dolphin-style processing.

Branches
--------
CURRENT_FINAL_C
    Existing production result.

STAGE7_SAVED_LEGACY_RAMP_SCN
    Existing phuw_sm2.ph_uw under the exact same saved Ramp+SCN corrections.

MINTPY_NO_LEGACY_RAMP_SCN
    MintPy traditional SBAS configuration:
        minNormVelocity = yes
        weightFunc = no

MINTPY_COH_LEGACY_RAMP_SCN
    Pixel-wise spatial coherence as the WLS weight.

MINTPY_VAR_LEGACY_RAMP_SCN
    Pixel-wise inverse DS phase variance from coherence, using MintPy's
    effective-look convention:
        L = round(RLOOKS * ALOOKS / 1.94)
    unless --ncorrlooks is explicitly provided.

Important
---------
This V6 is an INVERSION ISOLATION benchmark. It deliberately applies the same
saved ph_ramp + ph_scn to all branches so that only the network inversion changes.
A new branch is NOT promoted to production from this test alone. If a MintPy-style
branch wins, the next step is to rebuild SCLA/SCN self-consistently on that branch.

Official MintPy behavior reproduced mathematically
--------------------------------------------------
- min-norm interval velocity formulation;
- temporal-baseline design matrix B;
- WLS with sqrt(weight);
- rcond concept fixed at 1e-5 for exact-SVD spot checks;
- temporal coherence from unweighted phase residual;
- coherence weights:
    coh -> w = gamma
    var -> w = 1 / DS phase variance
- coherence epsilon = 0.05;
- effective looks fallback for Sentinel-1:
    L = round(ALOOKS * RLOOKS / 1.94), at least 1.

Computational implementation
----------------------------
For all 375,051 PS, V6 uses a vectorized banded Cholesky solver for the
pixel-specific weighted normal equations. This is mathematically the same WLS
objective but far faster than 375k calls to scipy.linalg.lstsq.

To protect against numerical mismatch, an exact scipy.linalg.lstsq(cond=1e-5)
spot audit is run on random PS for every branch. If the banded result differs
materially, the branch is flagged and cannot be considered valid.

The existing V3 coherence cache is reused:
    <dataset>/_pixel_wls_cache_v3/coherence_ps_ifg_float16.dat

This cache is only input coherence sampling; the abandoned V3 inversion itself
is not reused.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg, sparse
from scipy.io import loadmat
from scipy.spatial import cKDTree


# =============================================================================
# generic IO
# =============================================================================

def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def mat_is_hdf5(path: Path) -> bool:
    try:
        import h5py
        with h5py.File(path, "r"):
            return True
    except Exception:
        return False


def classic_var(path: Path, var: str):
    obj = loadmat(
        path,
        variable_names=[var],
        squeeze_me=False,
        struct_as_record=False,
    )
    if var not in obj:
        raise KeyError(f"{path}:{var} not found")
    return np.asarray(obj[var])


class MatrixReader:
    def __init__(self, path: Path, var: str, nrow: int, ncol: int):
        self.path = Path(path)
        self.var = var
        self.nrow = int(nrow)
        self.ncol = int(ncol)
        self.hdf5 = mat_is_hdf5(path)
        self._classic = None
        self._transposed = False

        if self.hdf5:
            import h5py
            with h5py.File(path, "r") as h:
                if var not in h:
                    raise KeyError(f"{path}:{var} not found")
                shape = tuple(int(v) for v in h[var].shape)
            if shape == (nrow, ncol):
                pass
            elif shape == (ncol, nrow):
                self._transposed = True
            else:
                raise ValueError(f"{path}:{var} shape={shape}")
        else:
            a = classic_var(path, var)
            if a.shape == (nrow, ncol):
                self._classic = a
            elif a.shape == (ncol, nrow):
                self._classic = a
                self._transposed = True
            else:
                raise ValueError(f"{path}:{var} shape={a.shape}")

    def rows(self, idx):
        idx = np.asarray(idx, np.int64)
        if self.hdf5:
            import h5py
            order = np.argsort(idx)
            sidx = idx[order]
            with h5py.File(self.path, "r") as h:
                ds = h[self.var]
                a = (
                    np.asarray(ds[sidx, :])
                    if not self._transposed
                    else np.asarray(ds[:, sidx]).T
                )
            inv = np.empty_like(order)
            inv[order] = np.arange(len(order))
            return a[inv]
        a = self._classic.T if self._transposed else self._classic
        return np.asarray(a[idx, :])

    def block(self, r0, r1):
        if self.hdf5:
            import h5py
            with h5py.File(self.path, "r") as h:
                ds = h[self.var]
                return (
                    np.asarray(ds[r0:r1, :])
                    if not self._transposed
                    else np.asarray(ds[:, r0:r1]).T
                )
        a = self._classic.T if self._transposed else self._classic
        return np.asarray(a[r0:r1, :])


def matlab_datenum_to_datetime(v):
    ordinal = int(v)
    frac = float(v) - ordinal
    return (
        dt.datetime.fromordinal(ordinal)
        + dt.timedelta(days=frac)
        - dt.timedelta(days=366)
    )


def local_xy(lon, lat, lon0=None, lat0=None):
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    if lon0 is None:
        lon0 = float(np.nanmedian(lon))
    if lat0 is None:
        lat0 = float(np.nanmedian(lat))
    R = 6371008.8
    x = np.deg2rad(lon-lon0)*R*math.cos(math.radians(lat0))
    y = np.deg2rad(lat-lat0)*R
    return x, y, lon0, lat0


def lonlat_to_xy(lon, lat, lon0, lat0):
    R = 6371008.8
    return (
        math.radians(lon-lon0)*R*math.cos(math.radians(lat0)),
        math.radians(lat-lat0)*R,
    )


# =============================================================================
# metadata
# =============================================================================

def load_metadata(dataset: Path):
    p = loadmat(
        dataset/"ps2.mat",
        variable_names=["n_ps", "n_image", "lonlat"],
        squeeze_me=True,
        struct_as_record=False,
    )
    n_ps = int(np.asarray(p["n_ps"]).reshape(-1)[0])
    n_image = int(np.asarray(p["n_image"]).reshape(-1)[0])
    ll = np.asarray(p["lonlat"], float)
    if ll.shape[0] != n_ps and ll.shape[1] == n_ps:
        ll = ll.T

    n = loadmat(
        dataset/"phuw_sm2.mat",
        variable_names=["ifgday_ix", "day"],
        squeeze_me=True,
        struct_as_record=False,
    )
    ix = np.asarray(n["ifgday_ix"])
    if ix.shape[0] == 2 and ix.shape[1] != 2:
        ix = ix.T
    day = np.asarray(n["day"], float).reshape(-1)
    dts = [matlab_datenum_to_datetime(v) for v in day]

    z = np.asarray(ix, float)
    if (
        np.min(z) >= 1
        and np.max(z) <= n_image
        and np.allclose(z, np.round(z))
    ):
        edges = np.round(z).astype(int) - 1
    elif (
        np.min(z) >= 0
        and np.max(z) < n_image
        and np.allclose(z, np.round(z))
    ):
        edges = np.round(z).astype(int)
    else:
        lookup = {int(round(v)): i for i, v in enumerate(day)}
        dates = [int(d.strftime("%Y%m%d")) for d in dts]
        lookup2 = {v: i for i, v in enumerate(dates)}
        edges = np.empty_like(z, dtype=int)
        for r in range(len(z)):
            for c in range(2):
                v = int(round(z[r, c]))
                if v in lookup:
                    edges[r, c] = lookup[v]
                elif v in lookup2:
                    edges[r, c] = lookup2[v]
                else:
                    raise RuntimeError(f"Cannot map ifgday_ix={v}")

    return n_ps, n_image, ll[:,0], ll[:,1], edges, day, dts


def exact_ref(final_dir: Path, n_ps: int):
    p = final_dir/"reference"/"reference_ps_indices_1based.txt"
    ids = np.asarray(np.loadtxt(p, dtype=np.int64)).reshape(-1)-1
    ids = np.unique(ids)
    if len(ids) < 10 or np.any(ids < 0) or np.any(ids >= n_ps):
        raise RuntimeError(f"Invalid exact reference: {p}")
    return ids


def find_final_c(dataset):
    c = sorted(
        [
            p for p in dataset.glob("final_C_fixed_reference_*")
            if (p/"velocity"/"final_C_velocity_points.npz").exists()
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not c:
        raise FileNotFoundError("Final-C product not found")
    return c[0]


# =============================================================================
# MintPy-style design matrix
# =============================================================================

def interval_years(dts):
    return np.asarray(
        [
            (dts[i+1]-dts[i]).total_seconds()/86400.0/365.25
            for i in range(len(dts)-1)
        ],
        dtype=np.float64,
    )


def build_B_velocity(edges, dts):
    """MintPy minNormVelocity design: IFG phase = sum(v_interval * dt_interval)."""
    n_int = len(dts)-1
    dt_yr = interval_years(dts)
    rows, cols, vals = [], [], []
    span = np.empty(len(edges), int)

    for e, (a,b) in enumerate(edges):
        a=int(a); b=int(b)
        if a < b:
            lo, hi, sgn = a, b, 1.0
        else:
            lo, hi, sgn = b, a, -1.0
        span[e] = hi-lo
        for j in range(lo,hi):
            rows.append(e); cols.append(j); vals.append(sgn*dt_yr[j])

    B = sparse.csr_matrix(
        (np.asarray(vals), (rows,cols)),
        shape=(len(edges), n_int),
    )
    return B, dt_yr, span


def contributors(edges, n_image, max_bw):
    span=np.abs(edges[:,1]-edges[:,0])
    bw=int(np.max(span)-1)
    if bw > max_bw:
        raise RuntimeError(f"bandwidth={bw} > --max-bandwidth={max_bw}")
    mats=[]
    n=n_image-1
    for d in range(bw+1):
        rr=[]; cc=[]
        for e,(a,b) in enumerate(edges):
            lo=min(int(a),int(b)); hi=max(int(a),int(b))
            if hi-lo <= d:
                continue
            for j in range(lo+d,hi):
                rr.append(e); cc.append(j-d)
        mats.append(
            sparse.csr_matrix(
                (np.ones(len(rr)),(rr,cc)),
                shape=(len(edges),n-d),
            )
        )
    return mats,bw


def normal_band_rhs(W, Y, B, contrib, bw, dt_yr, ridge_rel):
    W=np.asarray(W,np.float64)
    Y=np.asarray(Y,np.float64)
    batch=W.shape[0]
    n=B.shape[1]
    Ab=np.zeros((batch,bw+1,n),np.float64)
    WT=W.T
    for d,C in enumerate(contrib):
        s=(C.T@WT).T
        if d==0:
            coeff=dt_yr*dt_yr
        else:
            coeff=dt_yr[d:]*dt_yr[:-d]
        Ab[:,d,d:] = s*coeff[None,:]

    rhs=(B.T@(W*Y).T).T
    scale=np.nanmedian(Ab[:,0,:],axis=1)
    ridge=ridge_rel*np.maximum(scale,1e-12)
    Ab[:,0,:]+=ridge[:,None]
    return Ab,rhs


def solve_banded_batch(Ab,rhs):
    Ab=np.asarray(Ab,np.float64); rhs=np.asarray(rhs,np.float64)
    batch,bw1,n=Ab.shape
    bw=bw1-1
    L=np.zeros_like(Ab)
    bad=np.zeros(batch,bool)
    eps=1e-14

    for j in range(n):
        diag=Ab[:,0,j].copy()
        for d in range(1,min(bw,j)+1):
            diag-=L[:,d,j]**2
        bj=(~np.isfinite(diag))|(diag<=eps)
        bad|=bj
        diag=np.where(bj,eps,diag)
        L[:,0,j]=np.sqrt(diag)

        for i in range(j+1,min(n,j+bw+1)):
            d=i-j
            val=Ab[:,d,i].copy()
            k0=max(0,j-bw,i-bw)
            for k in range(k0,j):
                val-=L[:,i-k,i]*L[:,j-k,j]
            L[:,d,i]=val/L[:,0,j]

    z=np.empty_like(rhs)
    for j in range(n):
        val=rhs[:,j].copy()
        for d in range(1,min(bw,j)+1):
            val-=L[:,d,j]*z[:,j-d]
        z[:,j]=val/L[:,0,j]

    x=np.empty_like(rhs)
    for j in range(n-1,-1,-1):
        val=z[:,j].copy()
        for i in range(j+1,min(n,j+bw+1)):
            val-=L[:,i-j,i]*x[:,i]
        x[:,j]=val/L[:,0,j]

    x[bad]=np.nan
    return x,bad


def velocities_to_phase(V,dt_yr):
    X=np.zeros((len(V),len(dt_yr)+1),np.float64)
    X[:,1:]=np.cumsum(V*dt_yr[None,:],axis=1)
    return X


# =============================================================================
# MintPy coherence weights
# =============================================================================

def ds_phase_variance_lut(L: int, coh_step=0.005):
    """Independent implementation of the DS phase-variance PDF used by MintPy."""
    L=int(max(1,min(L,80)))
    n=int(1.0/coh_step)
    coh=np.linspace(coh_step,1.0,num=n,dtype=np.float64)-coh_step/2.0

    # MintPy's phase_variance_ds() uses phi_num == len(coherence).
    phi=np.linspace(-np.pi,np.pi,n,dtype=np.float64)[:,None]
    dphi=2*np.pi/n
    g=coh[None,:]
    beta=np.abs(g)*np.cos(phi)

    A=(1-g*g)**L/(2*np.pi)
    B=math.gamma(2*L-1)/(math.gamma(L)**2 * 2**(2*(L-1)))

    denom=1-beta*beta
    C=((2*L-1)*beta/(denom**(L+0.5)))*(np.pi/2+np.arcsin(beta))
    C+=1/(denom**L)

    Dsum=np.zeros_like(C)
    if L>1:
        for r in range(L-1):
            c1=math.gamma(L-0.5)/math.gamma(L-0.5-r)
            c2=math.gamma(L-1-r)/math.gamma(L-1)
            term=c1*c2*(1+(2*r+1)*beta*beta)/(denom**(r+2))
            Dsum+=term
        Dsum/=2*(L-1)

    pdf=A*(B*C+Dsum)
    var=np.sum((phi*phi)*pdf*dphi,axis=0)
    bad=var<=0
    if np.any(~bad):
        var[bad]=np.min(var[~bad])
    else:
        var[:]=np.finfo(np.float64).eps
    return coh,var


def coherence_to_weight(C,mode,L,epsilon=0.05):
    C=np.asarray(C,np.float64)
    C=np.where(np.isfinite(C),C,epsilon)
    C=np.maximum(C,epsilon)

    if mode=="COH":
        return C

    if mode=="VAR":
        coh_lut,var_lut=ds_phase_variance_lut(L)
        step=0.005
        cmin=coh_lut[0]
        cmax=coh_lut[-1]
        cc=np.clip(C,cmin,cmax)
        idx=((cc-cmin)/step).astype(np.int16)
        idx=np.clip(idx,0,len(var_lut)-1)
        return 1.0/var_lut[idx]

    if mode=="NO":
        return np.ones_like(C)

    raise ValueError(mode)


# =============================================================================
# solver / exact SVD audit
# =============================================================================

def solve_mode(Y,C,mode,B,contrib,bw,dt_yr,L,args):
    finite=np.isfinite(Y)
    if mode=="NO":
        W=finite.astype(np.float64)
    else:
        W=coherence_to_weight(C,mode,L,epsilon=0.05)
        W=np.where(finite,W,0.0)

    # Row-wise scaling changes no WLS solution, improves numerical dynamic range.
    med=np.median(np.where(W>0,W,np.nan),axis=1)
    med=np.where(np.isfinite(med)&(med>0),med,1.0)
    W=W/med[:,None]
    Y0=np.where(finite,Y,0.0)

    Ab,rhs=normal_band_rhs(W,Y0,B,contrib,bw,dt_yr,args.ridge_rel)
    V,bad=solve_banded_batch(Ab,rhs)
    X=velocities_to_phase(V,dt_yr)

    pred=(B@V.T).T
    resid=np.where(finite,Y-pred,np.nan)
    tc=np.abs(np.nansum(np.exp(1j*resid),axis=1)/np.maximum(np.sum(finite,axis=1),1))
    return X,tc,bad,W


def exact_svd_spot(Y,C,mode,Bdense,dt_yr,L,band_X,args):
    rows=[]
    for i in range(len(Y)):
        y=Y[i]
        valid=np.isfinite(y)
        if mode=="NO":
            w=np.ones(np.count_nonzero(valid))
        else:
            w=coherence_to_weight(
                C[i,valid][None,:],mode,L,epsilon=0.05
            ).reshape(-1)
        sw=np.sqrt(w)
        Bv=Bdense[valid,:]
        yv=y[valid]
        V=linalg.lstsq(
            Bv*sw[:,None],
            yv*sw,
            cond=1e-5,
        )[0]
        X=np.zeros(len(dt_yr)+1)
        X[1:]=np.cumsum(V*dt_yr)
        # Gauge same first-date=0 already.
        e=X-band_X[i]
        rows.append({
            "mode":mode,
            "spot_index":i,
            "median_abs_phase_difference_rad":float(np.median(np.abs(e))),
            "max_abs_phase_difference_rad":float(np.max(np.abs(e))),
        })
    return rows


# =============================================================================
# truth
# =============================================================================

def choose_vfield(gdf,preferred):
    geom=gdf.geometry.name
    cols=[c for c in gdf.columns if c!=geom]
    for c in cols:
        if c.lower()==preferred.lower():
            return c
    for c in cols:
        if np.issubdtype(gdf[c].dtype,np.number):
            if re.sub(r"[^a-z0-9]","",c.lower()) in ("v","vel","velocity","rate"):
                return c
    raise RuntimeError(f"truth field not found: {cols}")


def read_truth(path,preferred,scale,lon0,lat0,bbox,buffer_m):
    import geopandas as gpd
    gdf=gpd.read_file(path)
    if gdf.crs is not None:
        gdf=gdf.to_crs(4326)
    vf=choose_vfield(gdf,preferred)
    lon=np.asarray(gdf.geometry.x,float)
    lat=np.asarray(gdf.geometry.y,float)
    v=np.asarray(gdf[vf],float)*scale
    good=np.isfinite(lon)&np.isfinite(lat)&np.isfinite(v)
    lon,lat,v=lon[good],lat[good],v[good]
    x,y,_,_=local_xy(lon,lat,lon0,lat0)
    xmin,xmax,ymin,ymax=bbox
    m=(x>=xmin-buffer_m)&(x<=xmax+buffer_m)&(y>=ymin-buffer_m)&(y<=ymax+buffer_m)
    return x[m],y[m],v[m]


def unique_match(ps_xy,t_xy,match_m):
    tree=cKDTree(t_xy)
    dist,tidx=tree.query(ps_xy,k=1,workers=-1)
    within=np.isfinite(dist)&(dist<=match_m)
    p=np.flatnonzero(within); d=dist[within]; t=tidx[within]
    order=np.argsort(d)
    tt=t[order]
    _,first=np.unique(tt,return_index=True)
    keep=np.sort(order[first])
    return p[keep],t[keep]


def metrics(p,t):
    g=np.isfinite(p)&np.isfinite(t)
    p=np.asarray(p[g],float); t=np.asarray(t[g],float)
    e=p-t
    cc=float(np.corrcoef(p,t)[0,1]) if np.std(p)>0 and np.std(t)>0 else np.nan
    return {
        "n":len(e),
        "rmse_mm_yr":float(np.sqrt(np.mean(e**2))),
        "mae_mm_yr":float(np.mean(np.abs(e))),
        "correlation":cc,
        "pred_std_mm_yr":float(np.std(p)),
        "truth_std_mm_yr":float(np.std(t)),
        "pred_std_over_truth_std":float(np.std(p)/np.std(t)) if np.std(t)>0 else np.nan,
    }


def slope_coeff(dts,year):
    ids=np.asarray([i for i,d in enumerate(dts) if d.year==year],int)
    c=np.zeros(len(dts))
    t0=dts[ids[0]]
    t=np.asarray([(dts[i]-t0).total_seconds()/86400/365.2425 for i in ids])
    tc=t-np.mean(t)
    c[ids]=tc/np.sum(tc**2)
    return c


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        default="/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized",
    )
    p.add_argument("--truth-dir",default="")
    p.add_argument("--truth-field",default="v")
    p.add_argument("--truth-scale",type=float,default=1.0)
    p.add_argument("--match-m",type=float,default=200.0)
    p.add_argument("--truth-buffer-m",type=float,default=1500.0)
    p.add_argument("--reference-lon",type=float,default=117.21775055)
    p.add_argument("--reference-lat",type=float,default=38.40310669)
    p.add_argument("--reference-radius-m",type=float,default=500.0)

    p.add_argument("--rlooks",type=int,default=4)
    p.add_argument("--alooks",type=int,default=1)
    p.add_argument("--ncorrlooks",type=float,default=0.0)

    p.add_argument("--block-ps",type=int,default=2500)
    p.add_argument("--max-bandwidth",type=int,default=32)
    p.add_argument("--ridge-rel",type=float,default=1e-10)
    p.add_argument("--svd-spot-count",type=int,default=32)
    p.add_argument("--svd-max-diff-rad",type=float,default=1e-3)

    p.add_argument("--out",default="")
    p.add_argument("--seed",type=int,default=20260812)
    p.add_argument("--self-test",action="store_true")
    return p.parse_args()


def self_test():
    # Check DS variance is positive and decreases with coherence.
    c,v=ds_phase_variance_lut(2)
    assert np.all(v>0)
    assert v[-1] < v[0]

    # Exact synthetic minNormVelocity reconstruction.
    dts=[dt.datetime(2020,1,1)+dt.timedelta(days=12*i) for i in range(8)]
    edges=np.array([(i,j) for i in range(8) for j in range(i+1,min(8,i+4))],int)
    B,dt_yr,span=build_B_velocity(edges,dts)
    rng=np.random.default_rng(0)
    V=rng.normal(size=(5,7))
    Y=(B@V.T).T
    con,bw=contributors(edges,8,10)
    W=np.ones_like(Y)
    Ab,rhs=normal_band_rhs(W,Y,B,con,bw,dt_yr,1e-12)
    Vr,bad=solve_banded_batch(Ab,rhs)
    assert not np.any(bad)
    assert np.max(np.abs(V-Vr))<1e-6
    print("SELF-TEST: PASS")


# =============================================================================
# main
# =============================================================================

def main():
    args=parse_args()
    if args.self_test:
        self_test(); return

    t0=time.time()
    dataset=Path(args.dataset).resolve()
    truth_dir=Path(args.truth_dir).resolve() if args.truth_dir else dataset/"cangzhou"
    final_dir=find_final_c(dataset)
    stamp=dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out=Path(args.out).resolve() if args.out else dataset/"_audit"/f"mintpy_network_inversion_v6_{stamp}"
    out.mkdir(parents=True,exist_ok=True)

    n_ps,n_image,lon,lat,edges,day,dts=load_metadata(dataset)
    n_ifg=len(edges)
    print("="*80)
    print("V6 MintPy-style network inversion benchmark")
    print("="*80)
    print("n_ps:",n_ps,"n_image:",n_image,"n_ifg:",n_ifg)
    print("NO spatial filtering / NO IFG deletion / SAME 4:1 inputs")

    B,dt_yr,span=build_B_velocity(edges,dts)
    Bd=B.toarray()
    rank=int(np.linalg.matrix_rank(Bd))
    if rank != n_image-1:
        raise RuntimeError(f"B rank {rank}/{n_image-1}")
    con,bw=contributors(edges,n_image,args.max_bandwidth)
    print("B rank:",rank,"bandwidth:",bw)

    L=(
        int(max(1,round(args.ncorrlooks)))
        if args.ncorrlooks>0
        else int(max(1,round(args.rlooks*args.alooks/1.94)))
    )
    print("effective independent looks L:",L)

    # V3 coherence cache is retained deliberately even after code cleanup.
    cache=dataset/"_pixel_wls_cache_v3"/"coherence_ps_ifg_float16.dat"
    if not cache.exists():
        raise FileNotFoundError(
            f"coherence cache missing: {cache}\n"
            "Do not rebuild V3 inversion; only restore/build its coherence cache."
        )
    if cache.stat().st_size != n_ps*n_ifg*2:
        raise RuntimeError("coherence cache size mismatch")
    coh=np.memmap(cache,mode="r",dtype=np.float16,shape=(n_ps,n_ifg))

    obs=MatrixReader(dataset/"phuw2_gacos.mat","ph_uw",n_ps,n_ifg)
    stage7=MatrixReader(dataset/"phuw_sm2.mat","ph_uw",n_ps,n_image)
    ramp=MatrixReader(dataset/"scla2.mat","ph_ramp",n_ps,n_image)
    scn=MatrixReader(dataset/"uw_space_time.mat","ph_scn",n_ps,n_image)

    ref=exact_ref(final_dir,n_ps)

    # wavelength
    wavelength=0.0554657595
    phase_to_mm=-wavelength/(4*np.pi)*1000.0

    # truth
    x,y,lon0,lat0=local_xy(lon,lat)
    ps_xy=np.column_stack([x,y])
    bbox=(float(x.min()),float(x.max()),float(y.min()),float(y.max()))
    refx,refy=lonlat_to_xy(args.reference_lon,args.reference_lat,lon0,lat0)
    truth={}
    for year in (2021,2022,2023):
        tx,ty,tvraw=read_truth(
            truth_dir/f"result{year}.shp",
            args.truth_field,args.truth_scale,lon0,lat0,bbox,args.truth_buffer_m,
        )
        tree=cKDTree(np.column_stack([tx,ty]))
        rid=np.asarray(tree.query_ball_point([refx,refy],r=args.reference_radius_m),int)
        tref=float(np.nanmedian(tvraw[rid]))
        tv=tvraw-tref
        pidx,tidx=unique_match(ps_xy,np.column_stack([tx,ty]),args.match_m)
        truth[year]=dict(v=tv,pidx=pidx,tidx=tidx,ref=tref)
        print(year,"matched",len(pidx),"truth_ref",tref)

    # reference branch time series for exact common downstream correction
    Yref=np.asarray(obs.rows(ref),np.float64)
    Cref=np.asarray(coh[ref,:],np.float32)
    Rref=np.asarray(ramp.rows(ref),np.float64)
    Sref=np.asarray(scn.rows(ref),np.float64)
    STref=np.asarray(stage7.rows(ref),np.float64)

    modes=["NO","COH","VAR"]
    ref_ts={}
    svd_rows=[]

    # Solve exact branch reference.
    for mode in modes:
        Xref,tc,bad,W=solve_mode(
            Yref,Cref,mode,B,con,bw,dt_yr,L,args
        )
        ref_ts[mode]=np.nanmedian(Xref-Rref-Sref,axis=0)
        print("ref solved",mode,"bad",int(np.count_nonzero(bad)))

    stage7_ref=np.nanmedian(STref-Rref-Sref,axis=0)

    # SVD numerical spot audit.
    rng=np.random.default_rng(args.seed)
    spot=np.sort(rng.choice(n_ps,size=min(args.svd_spot_count,n_ps),replace=False))
    Ys=np.asarray(obs.rows(spot),np.float64)
    Cs=np.asarray(coh[spot,:],np.float32)

    for mode in modes:
        Xb,_,_,_=solve_mode(Ys,Cs,mode,B,con,bw,dt_yr,L,args)
        svd_rows.extend(exact_svd_spot(Ys,Cs,mode,Bd,dt_yr,L,Xb,args))

    write_csv(out/"01_exact_svd_spot_audit.csv",svd_rows)
    max_svd=max(r["max_abs_phase_difference_rad"] for r in svd_rows)
    numeric_ok=max_svd<=args.svd_max_diff_rad
    print("max exact-SVD vs banded diff:",max_svd,"numeric_ok:",numeric_ok)

    # Output velocities only; no new full time-series product.
    branch_names=[
        "MINTPY_NO_LEGACY_RAMP_SCN",
        "MINTPY_COH_LEGACY_RAMP_SCN",
        "MINTPY_VAR_LEGACY_RAMP_SCN",
        "STAGE7_SAVED_LEGACY_RAMP_SCN",
    ]
    pred={
        b:{y:np.full(n_ps,np.nan,np.float32) for y in (2021,2022,2023)}
        for b in branch_names
    }
    tc_out={
        m:np.full(n_ps,np.nan,np.float32)
        for m in modes
    }
    bad_count={m:0 for m in modes}

    coeff={y:slope_coeff(dts,y) for y in (2021,2022,2023)}

    for r0 in range(0,n_ps,args.block_ps):
        r1=min(n_ps,r0+args.block_ps)
        Y=np.asarray(obs.block(r0,r1),np.float64)
        C=np.asarray(coh[r0:r1,:],np.float32)
        R=np.asarray(ramp.block(r0,r1),np.float64)
        S=np.asarray(scn.block(r0,r1),np.float64)
        ST=np.asarray(stage7.block(r0,r1),np.float64)

        Xcorr={}
        for mode in modes:
            X,tc,bad,_=solve_mode(Y,C,mode,B,con,bw,dt_yr,L,args)
            bad_count[mode]+=int(np.count_nonzero(bad))
            tc_out[mode][r0:r1]=tc.astype(np.float32)
            Xcorr[mode]=X-R-S-ref_ts[mode][None,:]

        Xstage=ST-R-S-stage7_ref[None,:]

        for year in (2021,2022,2023):
            c=coeff[year]
            pred["MINTPY_NO_LEGACY_RAMP_SCN"][year][r0:r1]=(
                (Xcorr["NO"]@c)*phase_to_mm
            ).astype(np.float32)
            pred["MINTPY_COH_LEGACY_RAMP_SCN"][year][r0:r1]=(
                (Xcorr["COH"]@c)*phase_to_mm
            ).astype(np.float32)
            pred["MINTPY_VAR_LEGACY_RAMP_SCN"][year][r0:r1]=(
                (Xcorr["VAR"]@c)*phase_to_mm
            ).astype(np.float32)
            pred["STAGE7_SAVED_LEGACY_RAMP_SCN"][year][r0:r1]=(
                (Xstage@c)*phase_to_mm
            ).astype(np.float32)

        if r0==0 or r1==n_ps or r1%50000<args.block_ps:
            print("PS",r1,"/",n_ps)

        del Y,C,R,S,ST,Xcorr,Xstage
        gc.collect()

    # Existing Final-C.
    final_vel={}
    with np.load(
        final_dir/"velocity"/"final_C_velocity_points.npz",
        allow_pickle=False,
    ) as z:
        for year in (2021,2022,2023):
            final_vel[year]=np.asarray(z[f"velocity_{year}_mm_yr"],float)

    rows=[]; pooled=[]
    all_branches=branch_names+["CURRENT_FINAL_C"]
    for b in all_branches:
        pp=[];tt=[]
        for year in (2021,2022,2023):
            td=truth[year]
            vv=final_vel[year] if b=="CURRENT_FINAL_C" else pred[b][year]
            p=vv[td["pidx"]]; t=td["v"][td["tidx"]]
            m=metrics(p,t)
            rows.append({"branch":b,"year":year,**m})
            g=np.isfinite(p)&np.isfinite(t)
            pp.append(p[g]);tt.append(t[g])
        pm=metrics(np.concatenate(pp),np.concatenate(tt))
        pooled.append({"branch":b,**pm})

    write_csv(out/"02_truth_by_year.csv",rows)
    write_csv(out/"03_truth_pooled.csv",pooled)

    # Temporal coherence truth-error quartiles.
    qrows=[]
    branch_mode={
        "MINTPY_NO_LEGACY_RAMP_SCN":"NO",
        "MINTPY_COH_LEGACY_RAMP_SCN":"COH",
        "MINTPY_VAR_LEGACY_RAMP_SCN":"VAR",
    }
    for b,mode in branch_mode.items():
        for year in (2021,2022,2023):
            td=truth[year]
            q=tc_out[mode][td["pidx"]]
            p=pred[b][year][td["pidx"]]
            t=td["v"][td["tidx"]]
            good=np.isfinite(q)&np.isfinite(p)&np.isfinite(t)
            qg=q[good]; pg=p[good]; tg=t[good]
            qe=np.quantile(qg,[0,.25,.5,.75,1])
            for i in range(4):
                mm=(qg>=qe[i])&(qg<(qe[i+1]) if i<3 else qg<=qe[i+1])
                mt=metrics(pg[mm],tg[mm])
                qrows.append({
                    "branch":b,"year":year,"quartile":i+1,
                    "tc_min":float(qe[i]),"tc_max":float(qe[i+1]),**mt
                })
    write_csv(out/"04_temporal_coherence_truth_quartiles.csv",qrows)

    pmap={r["branch"]:r for r in pooled}
    current=pmap["CURRENT_FINAL_C"]["rmse_mm_yr"]
    best_new=min(
        [r for r in pooled if r["branch"].startswith("MINTPY_")],
        key=lambda r:r["rmse_mm_yr"]
    )
    improvement=(current-best_new["rmse_mm_yr"])/current

    classification=(
        "MINTPY_STYLE_INVERSION_PROMISING_REBUILD_DOWNSTREAM"
        if numeric_ok and improvement>0.02
        else
        "MINTPY_STYLE_INVERSION_NOT_ENOUGH_MOVE_UPSTREAM"
    )

    conclusion={
        "classification":classification,
        "official_style_settings":{
            "minNormVelocity":True,
            "rcond_exact_spot":1e-5,
            "weight_modes":["no","coh","var"],
            "coherence_epsilon":0.05,
            "effective_looks_L":L,
            "effective_looks_rule":(
                f"explicit NCORRLOOKS={args.ncorrlooks}" if args.ncorrlooks>0
                else f"round(RLOOKS*ALOOKS/1.94)=round({args.rlooks}*{args.alooks}/1.94)"
            ),
        },
        "network":{"n_ps":n_ps,"n_image":n_image,"n_ifg":n_ifg,"rank":rank,"bandwidth":bw},
        "numeric_spot_audit":{
            "max_phase_difference_rad":max_svd,
            "threshold_rad":args.svd_max_diff_rad,
            "pass":numeric_ok,
        },
        "bad_system_count":bad_count,
        "truth_pooled":pooled,
        "best_new_branch":best_new,
        "relative_improvement_best_new_vs_current_final_C":improvement,
        "important":(
            "All new inversion branches use the OLD saved Ramp+SCN only as an isolation "
            "test. No new branch is production-ready without re-estimating downstream "
            "SCLA/SCN self-consistently."
        ),
        "next_step_if_not_improved":(
            "Stop changing final velocity/inversion weights. Move upstream to a "
            "SARvey-like arc consistency / unwrap-error audit, then Dolphin-style "
            "SLC phase-linking ROI test."
        ),
        "runtime_seconds":time.time()-t0,
    }
    (out/"05_CONCLUSION.json").write_text(
        json.dumps(conclusion,ensure_ascii=False,indent=2),encoding="utf-8"
    )

    print("\nPOOLED")
    for r in pooled:
        print(r["branch"],"RMSE",r["rmse_mm_yr"],"corr",r["correlation"])
    print("\nclassification:",classification)
    print("output:",out)


if __name__=="__main__":
    main()
