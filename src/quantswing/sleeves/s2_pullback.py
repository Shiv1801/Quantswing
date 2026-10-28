"""S2 — Trend pullback (03 §4).

Setup: TQ120 in the top ``1 − tq_xs_quantile_min`` of the gated cross-section · R² ≥
``r2_frac_of_median`` × its ``r2_median_window`` median · DEV ∈ [dev_min, dev_max] ·
OUHL ≤ ``ouhl_max`` · knife filter: max VOLZ over the last ``knife_window`` < ``knife_volz_max``.
(Regime ≠ Bear/Crisis is applied by the P5 regime model.)
Trigger: Kalman slope t-stat crosses above ``kal_t_trigger`` within ``setup_memory``
sessions of a setup day → market-on-open next session.
"""

from __future__ import annotations

import pandas as pd

from quantswing.core.config import S2Cfg
from quantswing.factors.common import Panels
from quantswing.sleeves.base import FactorView, SleeveSignals, xs_pct_rank


def build(p: Panels, f: FactorView, gate: pd.DataFrame, c: S2Cfg) -> SleeveSignals:
    """S2 signals."""
    g = gate.astype(bool)
    r2 = f["R2_120"]
    r2_med = r2.rolling(c.r2_median_window, min_periods=c.r2_median_window // 2).median()
    dev = f["DEV"]
    setup = (
        (xs_pct_rank(f["TQ120"], g) >= c.tq_xs_quantile_min)
        & (r2 >= c.r2_frac_of_median * r2_med)
        & (dev >= c.dev_min)
        & (dev <= c.dev_max)
        & (f["OUHL"] <= c.ouhl_max)
        & (f["VOLZ"].rolling(c.knife_window, min_periods=1).max() < c.knife_volz_max)
        & g
    )
    recent = setup.astype(float).rolling(c.setup_memory, min_periods=1).max() > 0
    kt = f["KAL_T"]
    cross = (kt > c.kal_t_trigger) & (kt.shift(1) <= c.kal_t_trigger)
    return SleeveSignals("S2", recent & cross & g)
