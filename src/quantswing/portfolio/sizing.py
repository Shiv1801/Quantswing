"""Target-weight rules for the factor portfolio (04 §5–6 adapted to a ranked book, D29).

``w_i ∝ 1/σ_i`` (or equal), normalised to the gross budget, capped at ``single_name_max``
(excess redistributed), then scaled by ``min(1, σ_target / σ_forecast)`` using the
Ledoit–Wolf covariance, then by the regime multiplier and the drawdown multiplier.
"""

from __future__ import annotations

import numpy as np


def capped_normalise(raw: np.ndarray, gross: float, cap: float, iters: int = 20) -> np.ndarray:
    """Scale ``raw`` (≥0) to sum to ``gross`` with each weight ≤ ``cap`` (water-filling)."""
    w = np.where(np.isfinite(raw) & (raw > 0), raw, 0.0)
    if w.sum() <= 0:
        return w
    w = w / w.sum() * gross
    for _ in range(iters):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over & (w > 0)
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    out: np.ndarray = np.minimum(w, cap)
    return out


def vol_scale(weights: np.ndarray, cov: np.ndarray, target_vol: float) -> float:
    """``min(1, target / forecast)`` (1 if the forecast is unavailable)."""
    from quantswing.portfolio.covariance import portfolio_vol

    fv = portfolio_vol(weights, cov)
    return 1.0 if fv <= 0 or not np.isfinite(fv) else min(1.0, target_vol / fv)


def sector_capped(
    raw: np.ndarray,
    labels: np.ndarray,
    gross: float,
    name_cap: float,
    sector_cap: float,
    iters: int = 30,
) -> np.ndarray:
    """Weights ∝ ``raw`` summing to ≤ ``gross`` with each name ≤ ``name_cap`` and each
    sector's total ≤ ``sector_cap`` (``Unknown`` is not capped). Excess is redistributed
    to uncapped names; what cannot be placed stays in cash."""
    w = capped_normalise(raw, gross, name_cap)
    frozen = np.zeros(len(w), dtype=bool)
    for _ in range(iters):
        changed = False
        for g in np.unique(labels):
            if g == "Unknown":
                continue
            m = labels == g
            tot = w[m].sum()
            if tot > sector_cap + 1e-12:
                w[m] *= sector_cap / tot
                frozen |= m
                changed = True
        room = gross - w.sum()
        free = ~frozen & (w > 0) & (w < name_cap - 1e-12)
        if room > 1e-9 and free.any():
            add = room * w[free] / w[free].sum()
            w[free] = np.minimum(w[free] + add, name_cap)
            changed = True
        if not changed:
            break
    out: np.ndarray = w
    return out
