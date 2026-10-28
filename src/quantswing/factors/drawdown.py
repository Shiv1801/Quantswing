"""DD / RR — drawdown and recovery ratio, and H52 — 52-week-high proximity (03 §2).

* ``DD = P / ATH − 1``  (ATH over available history — Fyers depth, D22)
* ``RR = (P − trough) / (ATH − trough)``, trough = lowest close since the ATH (1 at the ATH)
* ``H52 = P / max(P over 252 sessions)``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantswing.factors.common import Panels


def drawdown_recovery(close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Causal DD and RR; NaN closes leave the running state unchanged."""
    x = close.to_numpy(dtype="float64")
    t_len, n = x.shape
    ath = np.full(n, np.nan)
    trough = np.full(n, np.nan)
    dd = np.full((t_len, n), np.nan)
    rr = np.full((t_len, n), np.nan)
    for t in range(t_len):
        p = x[t]
        has = ~np.isnan(p)
        new_high = has & (np.isnan(ath) | (p >= ath))
        ath = np.where(new_high, p, ath)
        trough = np.where(new_high, p, np.where(has, np.fmin(trough, p), trough))
        dd[t, has] = p[has] / ath[has] - 1.0
        span = ath - trough
        with np.errstate(divide="ignore", invalid="ignore"):
            rr_t = np.where(span > 0, (p - trough) / span, 1.0)
        rr[t, has] = rr_t[has]
    idx, cols = close.index, close.columns
    return pd.DataFrame(dd, index=idx, columns=cols), pd.DataFrame(rr, index=idx, columns=cols)


def high52(close: pd.DataFrame, n: int) -> pd.DataFrame:
    """``P / rolling max(P, n)``."""
    out: pd.DataFrame = close / close.rolling(n, min_periods=n).max()
    return out


def compute(p: Panels, h52_window: int) -> dict[str, pd.DataFrame]:
    """``{"DD": ..., "RR": ..., "H52": ...}``."""
    dd, rr = drawdown_recovery(p.close)
    return {"DD": dd, "RR": rr, "H52": high52(p.close, h52_window)}
