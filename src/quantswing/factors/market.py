"""BETA / IVOL vs the benchmark and RMOM — residual momentum (03 §2).

* ``BETA = cov_252(r, r_b) / var_252(r_b)``;  ``IVOL = sqrt(var(r) − β² var(r_b))`` (daily)
* ``RMOM = (Σ r − β Σ r_b) / (IVOL · √m)`` over sessions ``[t−k, t−skip)``, m = k − skip.
  Stand-in for RSEC: sector indices are not available point-in-time (D22), so relative
  strength is measured against the market benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantswing.factors.common import Panels


def beta_ivol(r: pd.DataFrame, rb: pd.Series, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling beta and idiosyncratic daily vol (full window of joint data required)."""
    rbm = pd.DataFrame(
        np.broadcast_to(rb.to_numpy()[:, None], r.shape), index=r.index, columns=r.columns
    )
    rbm = rbm.where(r.notna())
    m_r, m_b = r.rolling(n, min_periods=n).mean(), rbm.rolling(n, min_periods=n).mean()
    cov = (r * rbm).rolling(n, min_periods=n).mean() - m_r * m_b
    var_b = (rbm * rbm).rolling(n, min_periods=n).mean() - m_b * m_b
    var_r = (r * r).rolling(n, min_periods=n).mean() - m_r * m_r
    beta = cov / var_b.where(var_b > 0)
    ivar = (var_r - beta * beta * var_b).clip(lower=0)
    return beta, pd.DataFrame(np.sqrt(ivar), index=r.index, columns=r.columns)


def residual_momentum(
    r: pd.DataFrame, rb: pd.Series, beta: pd.DataFrame, ivol: pd.DataFrame, k: int, skip: int
) -> pd.DataFrame:
    """Beta-hedged cumulative return over ``[t−k, t−skip)`` in units of idiosyncratic σ."""
    m = k - skip
    sum_r = r.shift(skip).rolling(m, min_periods=m).sum()
    sum_b = rb.shift(skip).rolling(m, min_periods=m).sum()
    resid = sum_r.sub(beta.mul(sum_b, axis=0))
    out: pd.DataFrame = resid / (ivol.where(ivol > 0) * np.sqrt(m))
    return out


def compute(
    p: Panels, beta_window: int, rmom_window: int, rmom_skip: int
) -> dict[str, pd.DataFrame]:
    """``{"BETA": ..., "IVOL": ..., "RMOM": ...}``."""
    r, rb = p.r, p.rb
    beta, ivol = beta_ivol(r, rb, beta_window)
    return {
        "BETA": beta,
        "IVOL": ivol,
        "RMOM": residual_momentum(r, rb, beta, ivol, rmom_window, rmom_skip),
    }
