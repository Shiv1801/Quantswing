"""DEV — trend deviation and OUHL — mean-reversion half-life (03 §2).

* ``DEV = (ln P_t − fitted_t) / resid_sd`` from a trailing 120-session regression on time.
* ``OUHL``: AR(1) on DEV, ``x_t = a + β x_{t−1} + ε``;  ``HL = −ln 2 / ln β`` for 0 < β < 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantswing.factors.common import Panels, rolling_ar1_beta, rolling_time_regression


def trend_deviation(logp: pd.DataFrame, n: int) -> pd.DataFrame:
    """Standardised residual of ln P from its trailing time regression, at window end."""
    lr = rolling_time_regression(logp, n)
    out: pd.DataFrame = (logp - lr.fitted_end) / lr.resid_sd.where(lr.resid_sd > 0)
    return out


def ou_half_life(dev: pd.DataFrame, n: int) -> pd.DataFrame:
    """Half-life (sessions) of DEV's AR(1) mean reversion; NaN if not mean-reverting."""
    beta = rolling_ar1_beta(dev, n)
    ok = (beta > 0) & (beta < 1)
    out: pd.DataFrame = -np.log(2.0) / np.log(beta.where(ok))
    return out


def compute(p: Panels, dev_window: int, hl_window: int) -> dict[str, pd.DataFrame]:
    """``{"DEV": ..., "OUHL": ...}``."""
    dev = trend_deviation(p.logp, dev_window)
    return {"DEV": dev, "OUHL": ou_half_life(dev, hl_window)}
