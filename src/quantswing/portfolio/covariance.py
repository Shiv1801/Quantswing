"""Ledoit–Wolf shrunk covariance of daily returns (04 §6 vol forecast)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ledoit_wolf(returns: pd.DataFrame) -> np.ndarray:
    """Shrunk covariance (daily) of all columns; missing returns count as 0 (shape preserved)."""
    from sklearn.covariance import LedoitWolf

    r = returns.fillna(0.0).to_numpy(dtype="float64")
    if r.shape[0] < 3 or r.shape[1] == 0:
        return np.zeros((returns.shape[1], returns.shape[1]))
    cov = LedoitWolf().fit(r).covariance_
    return np.asarray(cov)


def portfolio_vol(weights: np.ndarray, cov: np.ndarray, ann: int = 252) -> float:
    """Annualised volatility of a weight vector."""
    v = float(weights @ cov @ weights)
    return float(np.sqrt(max(v, 0.0) * ann))
