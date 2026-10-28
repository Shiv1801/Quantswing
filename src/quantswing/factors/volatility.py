"""CMP — volatility compression and RNG — range tightness (03 §2).

* ``CMP = pct_rank_{250}(σ_10 / σ_60)``   (low = coiled)
* ``RNG = (max H_n − min L_n) / (P · σ_60 · √n)``, n = 20   (low = tight)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantswing.factors.common import Panels, rolling_sigma


def compression(r: pd.DataFrame, short: int, long: int, rank_window: int) -> pd.DataFrame:
    """Percentile rank (0–1] of ``σ_short/σ_long`` within its own trailing ``rank_window``."""
    ratio = rolling_sigma(r, short) / rolling_sigma(r, long)
    return ratio.rolling(rank_window, min_periods=rank_window).rank(pct=True)


def range_tightness(p: Panels, n: int, vol_window: int) -> pd.DataFrame:
    """Range of the last ``n`` sessions in units of expected √n-day move."""
    rng = p.high.rolling(n, min_periods=n).max() - p.low.rolling(n, min_periods=n).min()
    out: pd.DataFrame = rng / (p.close * rolling_sigma(p.r, vol_window) * np.sqrt(n))
    return out


def compute(
    p: Panels, short: int, long: int, rank_window: int, n: int, vol_window: int
) -> dict[str, pd.DataFrame]:
    """``{"CMP": ..., "RNG": ...}``."""
    return {
        "CMP": compression(p.r, short, long, rank_window),
        "RNG": range_tightness(p, n, vol_window),
    }
