"""p(win) calibration and expected value (03 §3.5–3.6).

* Logistic regression ``p = σ(a + b·x)`` of the triple-barrier outcome (label = +1) on a
  signal score x, fitted by Newton–Raphson (IRLS) with a small ridge penalty.
* ``EV_R = p·W̄ − (1−p)·L̄ − cost_R`` in R units, with W̄/L̄ the mean realised R of
  winning/losing training labels and ``cost_R`` = round-trip cost fraction / (R/P).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Logistic:
    """Fitted 1-D logistic model on a standardised score."""

    a: float
    b: float
    mu: float
    sd: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        """p(win) for raw scores ``x`` (NaN score → base rate)."""
        z = np.where(np.isfinite(x), (x - self.mu) / self.sd, 0.0)
        return np.asarray(1 / (1 + np.exp(-(self.a + self.b * z))))


def fit_logistic(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3, iters: int = 50) -> Logistic:
    """IRLS fit; degenerate inputs fall back to the base rate (b = 0)."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m].astype(float)
    base = float(np.clip(y.mean() if len(y) else 0.5, 1e-4, 1 - 1e-4))
    a0 = float(np.log(base / (1 - base)))
    if len(y) < 10 or x.std() == 0:
        return Logistic(a0, 0.0, float(x.mean()) if len(x) else 0.0, 1.0)
    mu, sd = float(x.mean()), float(x.std())
    xmat = np.column_stack([np.ones(len(x)), (x - mu) / sd])
    beta = np.array([a0, 0.0])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-xmat @ beta))
        w = p * (1 - p)
        grad = xmat.T @ (y - p) - ridge * np.r_[0.0, beta[1]]
        hess = (xmat.T * w) @ xmat + ridge * np.diag([0.0, 1.0])
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.abs(step).max() < 1e-8:
            break
    return Logistic(float(beta[0]), float(beta[1]), mu, sd)


@dataclass(frozen=True)
class EvModel:
    """p(win) model + payoff stats (R units)."""

    logit: Logistic
    avg_win_r: float
    avg_loss_r: float  # positive number
    n: int

    def ev(self, score: np.ndarray, cost_r: np.ndarray) -> np.ndarray:
        """Expected value in R per signal."""
        p = self.logit.predict(score)
        return np.asarray(p * self.avg_win_r - (1 - p) * self.avg_loss_r - cost_r, dtype="float64")


def fit_ev(score: np.ndarray, label: np.ndarray, ret_r: np.ndarray) -> EvModel:
    """Fit on training labels: ``label`` ∈ {−1, 0, +1}, ``ret_r`` realised R per label."""
    y = (label == 1).astype(float)
    lg = fit_logistic(score, y)
    wins, losses = ret_r[ret_r > 0], ret_r[ret_r <= 0]
    return EvModel(
        lg,
        float(wins.mean()) if len(wins) else 0.0,
        float(-losses.mean()) if len(losses) else 1.0,
        int(np.isfinite(score).sum()),
    )
