"""Factors that need NSE data (D31, 03 §2). Each is built only when its input is loaded.

* **DELZ** — delivery surprise: ``(deliv% − mean_n) / std_n`` over ``n`` = 60 sessions.
* **RSEC** — sector-relative strength: residual momentum after regressing the stock on its
  sector's return (equal-weight, leave-one-out) — ``(Σr − β Σr_sec) / (σ_ε √m)`` over
  ``[t−k, t−skip)``. Sectors with too few names fall back to the benchmark.
* **PLEDGE** — 1 if NSE flags high promoter pledge / encumbrance that day (QUAL-lite).
* **EARN** — earnings reaction (QUAL-lite): 2-session abnormal return ``Σ(r − β r_b)`` over
  the results session and the next, ÷ ``σ_ε √2``; known at the close of the second session
  and carried for ``hold_sessions``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from quantswing.core.config import FactorsCfg
from quantswing.factors.common import Panels
from quantswing.factors.market import beta_ivol

OTHER = "Other"
UNKNOWN = "Unknown"


def sector_labels(sector: pd.Series | None, cols: pd.Index, min_names: int) -> np.ndarray:
    """Per column: NSE sector, small sectors pooled into ``Other``, missing → ``Unknown``."""
    s = sector.reindex(cols) if sector is not None else pd.Series(index=cols, dtype=object)
    s = s.where(s.notna() & (s.astype(str).str.strip() != ""), UNKNOWN).astype(str)
    counts = s.value_counts()
    small = counts[counts < min_names].index.difference([UNKNOWN])
    return s.where(~s.isin(small), OTHER).to_numpy()


def delivery_surprise(deliv: pd.DataFrame, n: int) -> pd.DataFrame:
    """z-score of delivery % vs its own trailing ``n`` sessions."""
    m = deliv.rolling(n, min_periods=n).mean()
    sd = deliv.rolling(n, min_periods=n).std()
    out: pd.DataFrame = (deliv - m) / sd.where(sd > 0)
    return out


def sector_return(
    r: pd.DataFrame, labels: np.ndarray, rb: pd.Series, min_names: int
) -> pd.DataFrame:
    """Leave-one-out equal-weight sector return per column (benchmark where too few names)."""
    v = r.to_numpy(dtype="float64")
    valid = np.isfinite(v)
    x = np.where(valid, v, 0.0)
    out = np.broadcast_to(rb.to_numpy(dtype="float64")[:, None], v.shape).copy()
    for g in np.unique(labels):
        if g in (UNKNOWN,):
            continue
        cols = labels == g
        s = x[:, cols].sum(axis=1, keepdims=True)
        n = valid[:, cols].sum(axis=1, keepdims=True)
        own = x[:, cols]
        n_other = n - valid[:, cols]
        with np.errstate(invalid="ignore", divide="ignore"):
            loo = (s - own) / n_other
        out[:, cols] = np.where(n_other >= min_names, loo, out[:, cols])
    return pd.DataFrame(out, index=r.index, columns=r.columns)


def beta_ivol_vs(r: pd.DataFrame, rbm: pd.DataFrame, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling beta / idiosyncratic σ of each column vs its own benchmark column."""
    rbm = rbm.where(r.notna())
    m_r, m_b = r.rolling(n, min_periods=n).mean(), rbm.rolling(n, min_periods=n).mean()
    cov = (r * rbm).rolling(n, min_periods=n).mean() - m_r * m_b
    var_b = (rbm * rbm).rolling(n, min_periods=n).mean() - m_b * m_b
    var_r = (r * r).rolling(n, min_periods=n).mean() - m_r * m_r
    beta = cov / var_b.where(var_b > 0)
    ivar = (var_r - beta * beta * var_b).clip(lower=0)
    return beta, pd.DataFrame(np.sqrt(ivar), index=r.index, columns=r.columns)


def sector_momentum(
    r: pd.DataFrame, rs: pd.DataFrame, beta: pd.DataFrame, ivol: pd.DataFrame, k: int, skip: int
) -> pd.DataFrame:
    """RSEC: sector-hedged cumulative return over ``[t−k, t−skip)`` in σ_ε units."""
    m = k - skip
    sum_r = r.shift(skip).rolling(m, min_periods=m).sum()
    sum_s = rs.shift(skip).rolling(m, min_periods=m).sum()
    out: pd.DataFrame = (sum_r - beta * sum_s) / (ivol.where(ivol > 0) * np.sqrt(m))
    return out


def earnings_reaction(p: Panels, events: pd.DataFrame, beta_window: int, hold: int) -> pd.DataFrame:
    """EARN panel (NaN outside ``hold`` sessions after each event's second session)."""
    r, rb = p.r, p.rb
    beta, ivol = beta_ivol(r, rb, beta_window)
    ar = r.sub(beta.shift(1).mul(rb, axis=0))
    car = ar + ar.shift(-1)  # at row e: AR_e + AR_{e+1} (placed at e+1 below)
    idx, cols = r.index, r.columns
    pos = pd.Series(np.arange(len(idx)), index=idx)
    colpos = pd.Series(np.arange(len(cols)), index=cols)
    ev = events[events["symbol"].isin(cols) & events["event_date"].isin(idx)]
    out = np.full(r.shape, np.nan)
    if not ev.empty:
        ri = pos.loc[ev["event_date"]].to_numpy()
        ci = colpos.loc[ev["symbol"]].to_numpy()
        keep = ri + 1 < len(idx)
        ri, ci = ri[keep], ci[keep]
        sd = ivol.shift(1).to_numpy()[ri, ci]
        val = car.to_numpy()[ri, ci] / (sd * np.sqrt(2.0))
        out[ri + 1, ci] = val
    x = pd.DataFrame(out, index=idx, columns=cols)
    res: pd.DataFrame = x.ffill(limit=hold)
    return res


def compute(p: Panels, fc: FactorsCfg) -> dict[str, pd.DataFrame]:
    """Available NSE factors."""
    out: dict[str, pd.DataFrame] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if p.deliv is not None:
            out["DELZ"] = delivery_surprise(p.deliv, fc.delivery_surprise.window)
        if p.sector is not None and p.sector.notna().any():
            sm = fc.sector_momentum
            labels = sector_labels(p.sector, p.close.columns, fc.xsec.min_sector_names)
            rs = sector_return(p.r, labels, p.rb, fc.xsec.min_sector_names // 2 or 1)
            beta, ivol = beta_ivol_vs(p.r, rs, sm.beta_window)
            out["RSEC"] = sector_momentum(p.r, rs, beta, ivol, sm.window, sm.skip)
        if p.pledge is not None and p.pledge.notna().any().any():
            out["PLEDGE"] = p.pledge.where(p.close.notna())
        if p.events is not None and not p.events.empty:
            out["EARN"] = earnings_reaction(
                p, p.events, fc.beta.window, fc.earnings_reaction.hold_sessions
            )
    return out
