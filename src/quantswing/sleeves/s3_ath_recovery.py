"""S3 — Correction → recovery toward a new ATH (03 §4), tranche 1.

Setup: deepest drawdown over the last ``dd_lookback`` sessions in [dd_min, dd_max] ·
RR ≥ ``rr_min`` · HURST > ``hurst_min`` · Kalman slope accelerating (above its value
``kal_slope_accel_window`` sessions ago) with t-stat ≥ ``kal_t_min``.
Signal on the first day the setup holds; market-on-open next session.
Tranche 2 (pyramid on ATH break) is deferred (D27). DELZ is unavailable (D22).
"""

from __future__ import annotations

import pandas as pd

from quantswing.core.config import S3Cfg
from quantswing.factors.common import Panels
from quantswing.sleeves.base import FactorView, SleeveSignals, rising_edge


def build(p: Panels, f: FactorView, gate: pd.DataFrame, c: S3Cfg) -> SleeveSignals:
    """S3 signals."""
    g = gate.astype(bool)
    deepest = f["DD"].rolling(c.dd_lookback, min_periods=c.dd_lookback // 2).min()
    slope = f["KAL_SLOPE"]
    setup = (
        (deepest >= c.dd_min)
        & (deepest <= c.dd_max)
        & (f["RR"] >= c.rr_min)
        & (f["HURST"] > c.hurst_min)
        & (slope > slope.shift(c.kal_slope_accel_window))
        & (f["KAL_T"] >= c.kal_t_min)
        & g
    )
    return SleeveSignals("S3", rising_edge(setup))
