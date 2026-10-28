"""TQ — trend quality (03 §2).

Regress ``ln P`` on time over ``n`` sessions → slope ``b`` and ``R²``;
``TQ_n = (e^{252·b} − 1) × R²``  (annualised trend return weighted by trend cleanliness).
"""

from __future__ import annotations

import pandas as pd

from quantswing.factors.common import Panels, exp_df, rolling_time_regression


def trend_quality(logp: pd.DataFrame, n: int, ann: int) -> pd.DataFrame:
    """``(exp(ann·b) − 1) · R²`` from a trailing ``n``-session regression of ln P on t."""
    lr = rolling_time_regression(logp, n)
    out: pd.DataFrame = (exp_df(lr.slope * ann) - 1.0) * lr.r2
    return out


def compute(p: Panels, windows: list[int], ann: int) -> dict[str, pd.DataFrame]:
    """``{"TQ60": ..., "TQ120": ..., "R2_120": ...}`` (R² of the longest window, for S2)."""
    lp = p.logp
    out = {f"TQ{n}": trend_quality(lp, n, ann) for n in windows}
    out[f"R2_{max(windows)}"] = rolling_time_regression(lp, max(windows)).r2
    return out
