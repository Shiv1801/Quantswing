"""Market breadth for the regime model (03 §5): share of gated stocks whose trailing
``window``-session regression slope of ln P is positive. Point-in-time (trailing only)."""

from __future__ import annotations

import pandas as pd

from quantswing.factors.common import rolling_time_regression


def breadth(logp: pd.DataFrame, gate: pd.DataFrame, window: int, min_names: int) -> pd.Series:
    """Fraction (0–1) of gated names with a positive trend slope; NaN if too few names."""
    slope = rolling_time_regression(logp, window).slope.where(gate.astype(bool))
    n = slope.notna().sum(axis=1)
    frac = (slope > 0).sum(axis=1) / n.where(n > 0)
    out: pd.Series = frac.where(n >= min_names)
    return out
