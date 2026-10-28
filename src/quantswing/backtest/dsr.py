"""Deflated Sharpe Ratio (Bailey & López de Prado, 2014) — 05 §3 overfit check.

Probabilistic Sharpe:
``PSR(SR*) = Φ( (SR − SR*)·√(T−1) / √(1 − γ3·SR + (γ4−1)/4·SR²) )``
with per-period SR, skew γ3, (non-excess) kurtosis γ4, T observations.

Expected maximum Sharpe of N independent trials with variance ``V`` of trial Sharpes:
``SR* = √V · ((1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)))``, γ = Euler–Mascheroni.
DSR = PSR(SR*). N and V come from the run tracker (every logged trial counts).
"""

from __future__ import annotations

import math
from statistics import NormalDist

EULER_GAMMA = 0.5772156649015329
_N = NormalDist()


def psr(sr: float, sr_star: float, skew: float, kurt: float, t: int) -> float:
    """Probabilistic Sharpe ratio (all Sharpes per period, not annualised)."""
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr * sr
    if t < 2 or denom <= 0 or not math.isfinite(sr):
        return float("nan")
    return _N.cdf((sr - sr_star) * math.sqrt(t - 1) / math.sqrt(denom))


def expected_max_sharpe(n_trials: int, var_trial_sr: float) -> float:
    """Expected max per-period Sharpe among ``n_trials`` under the null."""
    if n_trials <= 1 or var_trial_sr <= 0:
        return 0.0
    a = _N.inv_cdf(1 - 1 / n_trials)
    b = _N.inv_cdf(1 - 1 / (n_trials * math.e))
    return math.sqrt(var_trial_sr) * ((1 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe(
    sr: float, skew: float, kurt: float, t: int, n_trials: int, var_trial_sr: float
) -> float:
    """DSR = PSR against the expected maximum Sharpe of the trials run so far."""
    return psr(sr, expected_max_sharpe(n_trials, var_trial_sr), skew, kurt, t)
