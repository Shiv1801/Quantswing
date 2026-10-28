"""S1 — Compression breakout (03 §4).

Setup (close t): CMP ≤ ``cmp_max`` · RNG in the bottom ``rng_xs_quantile_max`` of the
gated cross-section · TQ120 > ``tq_min`` · HURST > ``hurst_min``.
Trigger (t+1): high ≥ max(H over ``range_window``) + k·σ60·P (stop-entry), with
entry-day VOLZ ≥ ``volz_min``.
"""

from __future__ import annotations

import pandas as pd

from quantswing.core.config import S1Cfg
from quantswing.factors.common import Panels, rolling_sigma
from quantswing.sleeves.base import FactorView, SleeveSignals, xs_pct_rank


def build(
    p: Panels, f: FactorView, gate: pd.DataFrame, c: S1Cfg, sigma_window: int
) -> SleeveSignals:
    """S1 signals."""
    g = gate.astype(bool)
    setup = (
        (f["CMP"] <= c.cmp_max)
        & (xs_pct_rank(f["RNG"], g) <= c.rng_xs_quantile_max)
        & (f["TQ120"] > c.tq_min)
        & (f["HURST"] > c.hurst_min)
        & g
    )
    sig = rolling_sigma(p.r, sigma_window)
    level = (
        p.high.rolling(c.range_window, min_periods=c.range_window).max()
        + c.breakout_k_sigma * sig * p.close
    )
    return SleeveSignals(
        "S1", setup, entry_level=level.where(setup), confirm=f["VOLZ"] >= c.volz_min
    )
