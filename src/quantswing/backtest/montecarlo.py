"""Monte Carlo robustness (05 §3): trade-order reshuffles and bootstrap.

Each trade's return on equity-at-entry ``g_i`` is compounded in a random order
(reshuffle: permutation; bootstrap: resampling with replacement). The distribution of
max drawdown is compared with the realised one: pass if the 95th percentile ≤
``dd_mult_pass`` × backtest max DD.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class McResult:
    """Drawdown distribution summary."""

    dd_p50_reshuffle: float
    dd_p95_reshuffle: float
    dd_p95_bootstrap: float
    final_p05_bootstrap: float  # 5th percentile terminal wealth multiple


def _max_dd(paths: np.ndarray) -> np.ndarray:
    eq = np.cumprod(1 + paths, axis=1)
    eq = np.concatenate([np.ones((eq.shape[0], 1)), eq], axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    return np.asarray((eq / peak - 1).min(axis=1))


def simulate(trade_returns: np.ndarray, n_sims: int, seed: int) -> McResult:
    """Reshuffle + bootstrap max-DD percentiles (as positive fractions)."""
    g = np.asarray(trade_returns, dtype="float64")
    g = g[np.isfinite(g)]
    if len(g) < 2:
        nan = float("nan")
        return McResult(nan, nan, nan, nan)
    rng = np.random.default_rng(seed)
    perm = np.argsort(rng.random((n_sims, len(g))), axis=1)
    shuf = g[perm]
    boot = g[rng.integers(0, len(g), (n_sims, len(g)))]
    dd_s = -_max_dd(shuf)
    dd_b = -_max_dd(boot)
    final_b = np.prod(1 + boot, axis=1)
    return McResult(
        float(np.percentile(dd_s, 50)),
        float(np.percentile(dd_s, 95)),
        float(np.percentile(dd_b, 95)),
        float(np.percentile(final_b, 5)),
    )
