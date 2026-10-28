"""Exit parameter fitting from event paths (04 §3), per sleeve, on training data only.

For each signal (t, symbol): entry P0 = open of t+1, unit σ = σ60(t)·P0; track bars
t+1 … t+H. In σ units:
* MAE_h = −min(low − P0)/σ over the first ``winner_horizon`` bars (positive number)
* MFE   = max(high − P0)/σ over all H bars
* C_k   = (close_{t+k} − P0)/σ
Winners: C at ``winner_horizon`` > 0.

| Parameter | Rule |
|---|---|
| ``k_sl_sigma`` | pct(winners' MAE, ``winner_mae_pct``) × (1 + ``sl_buffer``), clipped |
| ``partial_at_r`` | MFE/k_sl level reached by ``partial_reach_frac`` of winners, clipped |
| ``trail_k_below_switch`` | pct(winners' giveback MFE − C_H, ``trail_giveback_pct``), clipped; above-switch k = ⅔ of it |
| ``time_stop_sessions`` | 2 × first k where mean C_k reaches ½ of its peak (signal half-life), clipped |
| ``max_hold_sessions`` | 2 × time stop, ≤ H |
Too few events (< ``min_events``) → config defaults, flagged low-confidence.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantswing.core.config import ExitFitCfg, ExitsCfg


@dataclass(frozen=True)
class EventPaths:
    """Per-event path statistics (σ units)."""

    mae_w: np.ndarray  # MAE within winner horizon
    mfe: np.ndarray
    close_k: np.ndarray  # events × H
    winners: np.ndarray  # bool


def event_paths(
    events: pd.DataFrame,
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    sigma: pd.DataFrame,
    horizon: int,
    winner_horizon: int,
) -> EventPaths:
    """Vectorised path extraction (loops over the horizon only)."""
    idx = close.index
    pos = pd.Series(np.arange(len(idx)), index=idx)
    col = pd.Series(np.arange(close.shape[1]), index=close.columns)
    ev = events[events["date"].isin(idx) & events["symbol"].isin(col.index)]
    r_i = pos.loc[ev["date"]].to_numpy()
    c_i = col.loc[ev["symbol"]].to_numpy()
    t_len = len(idx)
    ok = r_i + 1 < t_len
    r_i, c_i = r_i[ok], c_i[ok]
    op, hi, lw, cl = (x.to_numpy(dtype="float64") for x in (open_, high, low, close))
    p0 = op[r_i + 1, c_i]
    unit = sigma.to_numpy(dtype="float64")[r_i, c_i] * p0
    n = len(r_i)
    hmat = np.full((n, horizon), np.nan)
    lmat = np.full((n, horizon), np.nan)
    cmat = np.full((n, horizon), np.nan)
    for k in range(1, horizon + 1):
        rows = r_i + k
        inb = rows < t_len
        rr = np.where(inb, rows, t_len - 1)
        hmat[:, k - 1] = np.where(inb, (hi[rr, c_i] - p0) / unit, np.nan)
        lmat[:, k - 1] = np.where(inb, (lw[rr, c_i] - p0) / unit, np.nan)
        cmat[:, k - 1] = np.where(inb, (cl[rr, c_i] - p0) / unit, np.nan)
    cmat = pd.DataFrame(cmat).ffill(axis=1).to_numpy()  # carry through missing bars
    valid = np.isfinite(unit) & (unit > 0) & np.isfinite(p0)
    with warnings.catch_warnings():  # events at the data edge have no forward bars
        warnings.simplefilter("ignore", RuntimeWarning)
        mae_w = -np.nanmin(lmat[:, :winner_horizon], axis=1)
        mfe = np.nanmax(hmat, axis=1)
    c_w = cmat[:, winner_horizon - 1]
    keep = valid & np.isfinite(mae_w) & np.isfinite(mfe) & np.isfinite(c_w)
    return EventPaths(mae_w[keep], mfe[keep], cmat[keep], (c_w > 0)[keep])


@dataclass(frozen=True)
class FitResult:
    """Fitted exits + diagnostics."""

    exits: ExitsCfg
    n_events: int
    n_winners: int
    confident: bool
    half_life: float


def fit_exits(paths: EventPaths, fc: ExitFitCfg, default: ExitsCfg) -> FitResult:
    """Derive exit parameters from event paths (module table)."""
    n, w = len(paths.mfe), int(paths.winners.sum())
    if n < fc.min_events or w < max(10, fc.min_events // 5):
        return FitResult(default, n, w, False, float("nan"))
    win = paths.winners
    k_sl = float(np.percentile(np.maximum(paths.mae_w[win], 0), fc.winner_mae_pct)) * (
        1 + fc.sl_buffer
    )
    k_sl = float(np.clip(k_sl, *fc.k_sl_bounds))
    mfe_r = paths.mfe[win] / k_sl
    partial = float(np.quantile(mfe_r, 1 - fc.partial_reach_frac))
    partial = float(np.clip(partial, *fc.partial_r_bounds))
    last = paths.close_k[win, -1]
    giveback = paths.mfe[win] - last
    k_trail = float(
        np.clip(
            np.percentile(giveback[np.isfinite(giveback)], fc.trail_giveback_pct),
            *fc.trail_k_bounds,
        )
    )
    k_above = float(np.clip(k_trail * 2 / 3, fc.trail_k_bounds[0], k_trail))
    mean_c = np.nanmean(paths.close_k, axis=0)
    peak = float(np.nanmax(mean_c))
    if peak > 0:
        half = float(np.argmax(mean_c >= peak / 2) + 1)
        tstop = int(np.clip(2 * half, *fc.time_stop_bounds))
    else:
        half, tstop = float("nan"), default.time_stop_sessions
    max_hold = int(min(max(2 * tstop, tstop + 1), fc.horizon))
    ex = default.model_copy(
        update={
            "k_sl_sigma": k_sl,
            "partial_at_r": partial,
            "trail_k_below_switch": k_trail,
            "trail_k_above_switch": k_above,
            "time_stop_sessions": tstop,
            "max_hold_sessions": max_hold,
        }
    )
    return FitResult(ex, n, w, True, half)
