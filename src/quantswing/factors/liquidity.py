"""ILLIQ (Amihud), VOLZ (volume surprise) and SIZE proxy (03 §2–3).

* ``ILLIQ = −ln( mean_20( |r| / TV_crore ) )``   (higher = more liquid)
* ``VOLZ  = (ln V_t − mean_{t−50..t−1} ln V) / std_{t−50..t−1} ln V``
* ``SIZE  = ln(median_60 TV)`` — neutralisation control (market-cap proxy)
"""

from __future__ import annotations

import pandas as pd

from quantswing.factors.common import Panels, log_df


def amihud(r: pd.DataFrame, turnover: pd.DataFrame, n: int, crore: float) -> pd.DataFrame:
    """``−ln(mean_n(|r| / (TV / crore)))``; zero-turnover days excluded."""
    tv = (turnover / crore).where(turnover > 0)
    illiq = (r.abs() / tv).rolling(n, min_periods=n).mean()
    return -log_df(illiq)


def volume_surprise(volume: pd.DataFrame, n: int) -> pd.DataFrame:
    """z-score of today's ln volume against the previous ``n`` sessions."""
    lv = log_df(volume)
    prev = lv.shift(1).rolling(n, min_periods=n)
    mu, sd = prev.mean(), prev.std()
    out: pd.DataFrame = (lv - mu) / sd.where(sd > 0)
    return out


def size_proxy(turnover: pd.DataFrame, n: int) -> pd.DataFrame:
    """``ln(median_n turnover)``."""
    return log_df(turnover.where(turnover > 0).rolling(n, min_periods=n).median())


def compute(
    p: Panels, illiq_window: int, volz_window: int, crore: float
) -> dict[str, pd.DataFrame]:
    """``{"ILLIQ": ..., "VOLZ": ...}``."""
    return {
        "ILLIQ": amihud(p.r, p.turnover, illiq_window, crore),
        "VOLZ": volume_surprise(p.volume, volz_window),
    }
