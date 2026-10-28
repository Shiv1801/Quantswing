"""HURST — persistence via the variance ratio (03 §2).

``VR(q) = Var(r_q) / (q · Var(r_1))`` over a trailing window (r_q = overlapping q-day
returns). Since ``VR(q) ≈ q^{2H−1}``:  ``H = ½ · (1 + ln VR / ln q)``; H > 0.5 trending.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantswing.factors.common import Panels


def variance_ratio_hurst(r: pd.DataFrame, window: int, q: int) -> pd.DataFrame:
    """Hurst exponent implied by the rolling variance ratio."""
    rq = r.rolling(q, min_periods=q).sum()
    v1 = r.rolling(window, min_periods=window).var()
    vq = rq.rolling(window, min_periods=window).var()
    vr = vq / (q * v1.where(v1 > 0))
    out: pd.DataFrame = 0.5 * (1.0 + np.log(vr.where(vr > 0)) / np.log(q))
    return out


def compute(p: Panels, window: int, q: int) -> dict[str, pd.DataFrame]:
    """``{"HURST": ...}``."""
    return {"HURST": variance_ratio_hurst(p.r, window, q)}
