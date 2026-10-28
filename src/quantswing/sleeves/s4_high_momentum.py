"""S4 — 52-week / ATH momentum (03 §4; George–Hwang).

Setup: H52 ≥ ``h52_min`` · MOM126 in the top ``1 − mom_xs_quantile_min`` of the gated
cross-section · RMOM > ``rmom_min`` (RSEC stand-in, D24).
Trigger (close t): new closing high vs the prior ``high_window`` sessions with
VOLZ ≥ ``volz_min`` → market-on-open next session.
"""

from __future__ import annotations

import pandas as pd

from quantswing.core.config import S4Cfg
from quantswing.factors.common import Panels
from quantswing.sleeves.base import FactorView, SleeveSignals, xs_pct_rank


def build(p: Panels, f: FactorView, gate: pd.DataFrame, c: S4Cfg) -> SleeveSignals:
    """S4 signals."""
    g = gate.astype(bool)
    prior_high = p.close.shift(1).rolling(c.high_window, min_periods=c.high_window).max()
    sig = (
        (f["H52"] >= c.h52_min)
        & (xs_pct_rank(f["MOM126"], g) >= c.mom_xs_quantile_min)
        & (f["RMOM"] > c.rmom_min)
        & (p.close > prior_high)
        & (f["VOLZ"] >= c.volz_min)
        & g
    )
    return SleeveSignals("S4", sig)
