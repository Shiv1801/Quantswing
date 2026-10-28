"""Cross-sectional processing (03 §3, steps 1–3), per date across gated symbols.

1. Winsorise at the ``winsor_lower`` / ``winsor_upper`` cross-sectional quantiles.
2. z-score: ``(x − mean) / std``.
3. Neutralise: residual of z on ``[sector dummies, size]`` (OLS per date) → ``neutral_z``,
   re-standardised. Sector fixed effects via within-sector demeaning (Frisch–Waugh); NSE
   sectors (D31) are the current classification, not point-in-time (bias documented).
Dates with fewer than ``min_names_per_date`` valid names are set to NaN.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from quantswing.core.config import XsecCfg


def winsorise(x: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    """Clip each row to its [lower, upper] quantiles."""
    lo = x.quantile(lower, axis=1)
    hi = x.quantile(upper, axis=1)
    return x.clip(lower=lo, upper=hi, axis=0)


def zscore(x: pd.DataFrame) -> pd.DataFrame:
    """Row-wise standardisation."""
    mu = x.mean(axis=1)
    sd = x.std(axis=1)
    return x.sub(mu, axis=0).div(sd.where(sd > 0), axis=0)


def neutralise(z: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
    """Residual of ``z`` on ``[1, control]`` per row (joint-valid names only), re-z-scored."""
    c = control.where(z.notna())
    zz = z.where(c.notna())
    mz, mc = zz.mean(axis=1), c.mean(axis=1)
    cov = (zz.sub(mz, axis=0) * c.sub(mc, axis=0)).mean(axis=1)
    var = c.sub(mc, axis=0).pow(2).mean(axis=1)
    b = cov / var.where(var > 0)
    resid = zz.sub(mz, axis=0).sub(c.sub(mc, axis=0).mul(b, axis=0))
    return zscore(resid)


def demean_by_group(x: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Subtract each row's group mean (NaNs ignored) from every member of the group."""
    v = x.to_numpy(dtype="float64")
    out = np.full_like(v, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for g in np.unique(labels):
            cols = labels == g
            sub = v[:, cols]
            out[:, cols] = sub - np.nanmean(sub, axis=1, keepdims=True)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def neutralise_sector(
    z: pd.DataFrame, control: pd.DataFrame | None, labels: np.ndarray
) -> pd.DataFrame:
    """Residual of ``z`` on sector dummies (+ ``control``), re-z-scored (FWL)."""
    if control is not None:
        c = control.where(z.notna())
        zz = z.where(c.notna())
    else:
        c, zz = None, z
    zd = demean_by_group(zz, labels)
    if c is None:
        return zscore(zd)
    cd = demean_by_group(c, labels)
    cov = (zd * cd).mean(axis=1)
    var = (cd * cd).mean(axis=1)
    b = cov / var.where(var > 0)
    return zscore(zd.sub(cd.mul(b, axis=0)))


def process(
    raw: pd.DataFrame,
    gate: pd.DataFrame,
    size: pd.DataFrame | None,
    cfg: XsecCfg,
    sectors: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``raw`` → (``z``, ``neutral_z``) on the gated universe.

    ``sectors``: one label per column (see :func:`nse_factors.sector_labels`) or ``None``.
    """
    x = raw.where(gate)
    enough = x.notna().sum(axis=1) >= cfg.min_names_per_date
    x = x.where(enough, axis=0)
    z = zscore(winsorise(x, cfg.winsor_lower, cfg.winsor_upper))
    ctrl = size if cfg.neutralise_size else None
    if cfg.neutralise_sector and sectors is not None and len(np.unique(sectors)) > 1:
        nz = neutralise_sector(z, ctrl, sectors)
    else:
        nz = neutralise(z, ctrl) if ctrl is not None else z
    return z, nz
