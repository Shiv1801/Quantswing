"""Information coefficient analysis (05 §3 "Factor IC"; 03 §3.4 input).

* Forward return (TARGET ONLY, never a feature): ``fwd_h(t) = ln P_{t+h} − ln P_t``.
* Daily rank IC: Spearman correlation across names between factor(t) and fwd_h(t),
  computed as Pearson on cross-sectional ranks, joint-valid names only.
* Summary per horizon h: mean, std, IR = mean/std, hit rate (IC > 0), and a t-stat on
  NON-OVERLAPPING observations (every h-th date) — overlapping h-day returns inflate naive t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def forward_log_return(logp: pd.DataFrame, h: int) -> pd.DataFrame:
    """``ln P_{t+h} − ln P_t`` (uses future data by design: evaluation target only)."""
    return logp.shift(-h) - logp


def rank_ic(factor: pd.DataFrame, fwd: pd.DataFrame, min_names: int) -> pd.Series:
    """Daily cross-sectional Spearman IC."""
    joint = factor.notna() & fwd.notna()
    a = factor.where(joint).rank(axis=1)
    b = fwd.where(joint).rank(axis=1)
    a = a.sub(a.mean(axis=1), axis=0)
    b = b.sub(b.mean(axis=1), axis=0)
    num = (a * b).sum(axis=1)
    den = ((a * a).sum(axis=1) * (b * b).sum(axis=1)).pow(0.5)
    ic: pd.Series = num / den.where(den > 0)
    out: pd.Series = ic.where(joint.sum(axis=1) >= min_names)
    return out


@dataclass(frozen=True)
class IcStats:
    """Summary of a daily IC series at horizon ``h``."""

    h: int
    mean: float
    std: float
    ir: float
    t_nonoverlap: float
    hit: float
    n: int


def summarise(ic: pd.Series, h: int) -> IcStats:
    """Mean/std/IR/hit on all days; t-stat on every h-th day."""
    s = ic.dropna()
    if len(s) < 3:
        return IcStats(
            h, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), len(s)
        )
    sub = s.iloc[::h]
    t = (
        float(sub.mean() / sub.std() * np.sqrt(len(sub)))
        if len(sub) > 2 and sub.std() > 0
        else float("nan")
    )
    return IcStats(
        h=h,
        mean=float(s.mean()),
        std=float(s.std()),
        ir=float(s.mean() / s.std()) if s.std() > 0 else float("nan"),
        t_nonoverlap=t,
        hit=float((s > 0).mean()),
        n=len(s),
    )


def ic_by_year(ic: pd.Series) -> pd.Series:
    """Mean IC per calendar year."""
    s = ic.dropna()
    return s.groupby(pd.DatetimeIndex(s.index).year).mean()
