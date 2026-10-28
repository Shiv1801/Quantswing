"""MOM — volatility-adjusted momentum, skipping the last month (03 §2).

``MOM_k = (ln P_{t−skip} − ln P_{t−k}) / (σ_k · √k)``, σ_k = std of daily log returns over k.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantswing.factors.common import Panels, rolling_sigma


def vol_adjusted_momentum(logp: pd.DataFrame, r: pd.DataFrame, k: int, skip: int) -> pd.DataFrame:
    """``(lnP.shift(skip) − lnP.shift(k)) / (σ_k √k)``."""
    sig = rolling_sigma(r, k)
    out: pd.DataFrame = (logp.shift(skip) - logp.shift(k)) / (sig * np.sqrt(k))
    return out


def compute(p: Panels, lookbacks: list[int], skip: int) -> dict[str, pd.DataFrame]:
    """``{"MOM63": ..., "MOM126": ..., "MOM252": ...}``."""
    lp, r = p.logp, p.r
    return {f"MOM{k}": vol_adjusted_momentum(lp, r, k, skip) for k in lookbacks}
