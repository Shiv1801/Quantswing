"""4-state Gaussian HMM regime model (03 §5), strictly point-in-time.

Features (daily): benchmark log return, 20d realised vol, breadth, 5d change in ln(VIX).
* Refit monthly on an expanding window (first fit after ``min_train_sessions``).
* Features standardised with the training window's mean/std only.
* State probabilities are FILTERED (forward algorithm) — never smoothed with future data.
* States labelled after each fit: highest-vol state = Crisis; of the rest, lowest mean
  return = Bear; of the remaining two, higher vol = Bull-volatile, other = Bull-calm.
* Smoothing: the active state changes only after the new state's probability exceeds
  ``switch_prob`` for ``switch_days`` consecutive sessions.
Exposure multiplier per state from ``risk.regime_exposure_mult``.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantswing.core.config import RegimeCfg

STATES: tuple[str, ...] = ("bull_calm", "bull_volatile", "bear", "crisis")


def features(
    bench: pd.Series, vix: pd.Series | None, breadth: pd.Series, cfg: RegimeCfg
) -> pd.DataFrame:
    """Daily feature matrix (NaN rows dropped later)."""
    r = np.log(bench).diff()
    out = pd.DataFrame(
        {
            "ret": r,
            "rv20": r.rolling(cfg.rv_window, min_periods=cfg.rv_window).std() * np.sqrt(252),
            "breadth": breadth,
        }
    )
    if vix is not None and vix.notna().sum() > 0:
        out["dvix"] = np.log(vix.where(vix > 0)).diff(cfg.vix_change_window)
    else:
        out["dvix"] = 0.0
    out = out[[c for c in cfg.features if c in out]]
    return out.loc[
        :, out.notna().any()
    ]  # an input with no data at all (e.g. tiny universe) is skipped


@dataclass(frozen=True)
class FittedHmm:
    """Parameters of one fit (in standardised feature space) + state labels."""

    mu: np.ndarray  # scaler mean
    sd: np.ndarray  # scaler std
    means: np.ndarray  # k × d
    vars: np.ndarray  # k × d (diagonal)
    trans: np.ndarray  # k × k
    start: np.ndarray  # k
    labels: tuple[str, ...]  # label of each hidden state


def fit(x: np.ndarray, cfg: RegimeCfg) -> FittedHmm:
    """Fit a diagonal Gaussian HMM with hmmlearn and label the states."""
    from hmmlearn.hmm import GaussianHMM

    x = x[np.isfinite(x).all(axis=1)]
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    z = (x - mu) / sd
    m = GaussianHMM(
        n_components=cfg.n_states, covariance_type="diag", n_iter=cfg.n_iter, random_state=cfg.seed
    )
    hmm_log = logging.getLogger("hmmlearn")
    level = hmm_log.level
    hmm_log.setLevel(logging.CRITICAL)  # convergence chatter is expected on monthly refits
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(z)
    finally:
        hmm_log.setLevel(level)
    means = np.asarray(m.means_)
    vars_ = np.asarray([np.diag(c) for c in m.covars_])
    raw_means = means * sd + mu
    ret_i = 0  # feature 0 = ret
    vol_i = 1 if x.shape[1] > 1 else 0  # feature 1 = rv20
    order = list(range(cfg.n_states))
    crisis = max(order, key=lambda k: raw_means[k, vol_i])
    rest = [k for k in order if k != crisis]
    bear = min(rest, key=lambda k: raw_means[k, ret_i])
    rest = [k for k in rest if k != bear]
    bull_vol = max(rest, key=lambda k: raw_means[k, vol_i])
    labels = ["bull_calm"] * cfg.n_states
    labels[crisis], labels[bear], labels[bull_vol] = "crisis", "bear", "bull_volatile"
    return FittedHmm(
        mu, sd, means, vars_, np.asarray(m.transmat_), np.asarray(m.startprob_), tuple(labels)
    )


def filtered_probs(f: FittedHmm, x: np.ndarray) -> np.ndarray:
    """Forward-algorithm filtered state probabilities p(s_t | x_1..t) (T × k)."""
    z = (x - f.mu) / f.sd
    k = len(f.start)
    logb = -0.5 * (
        ((z[:, None, :] - f.means[None]) ** 2) / f.vars[None] + np.log(2 * np.pi * f.vars[None])
    ).sum(axis=2)
    out = np.empty((len(z), k))
    alpha = f.start.copy()
    for t in range(len(z)):
        pred = alpha if t == 0 else alpha @ f.trans
        b = np.exp(logb[t] - logb[t].max())
        a = pred * b
        s = a.sum()
        alpha = a / s if s > 0 else np.full(k, 1.0 / k)
        out[t] = alpha
    return out


def smooth_states(prob: pd.DataFrame, switch_prob: float, switch_days: int) -> pd.Series:
    """Active state with the p > switch_prob for switch_days consecutive days rule."""
    cur: str | None = None
    streak_state: str | None = None
    streak = 0
    out = []
    for _, row in prob.iterrows():
        if row.isna().all():
            out.append(cur)
            continue
        top = str(row.idxmax())
        if row[top] > switch_prob and top != cur:
            streak = streak + 1 if top == streak_state else 1
            streak_state = top
            if cur is None or streak >= switch_days:
                cur, streak = top, 0
        else:
            streak, streak_state = 0, None
        out.append(cur)
    return pd.Series(out, index=prob.index, name="state")


def run_regime(feat: pd.DataFrame, cfg: RegimeCfg, mult: dict[str, float]) -> pd.DataFrame:
    """Monthly-refit, filtered regime series: probs per labelled state, state, multiplier."""
    f = feat.dropna()
    idx = pd.DatetimeIndex(f.index)
    months = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).min().sort_values()
    probs = pd.DataFrame(np.nan, index=idx, columns=list(STATES))
    x_all = f.to_numpy(dtype="float64")
    starts = [int(idx.searchsorted(m0)) for m0 in months]
    arr = probs.to_numpy(copy=True)
    for k, pos0 in enumerate(starts):
        if pos0 < cfg.min_train_sessions:
            continue
        fitted = fit(x_all[:pos0], cfg)
        pos1 = starts[k + 1] if k + 1 < len(starts) else len(idx)
        pr = filtered_probs(fitted, x_all[:pos1])[pos0:pos1]
        for j, lab in enumerate(fitted.labels):
            c = STATES.index(lab)
            arr[pos0:pos1, c] = np.nan_to_num(arr[pos0:pos1, c], nan=0.0) + pr[:, j]
    probs = pd.DataFrame(arr, index=idx, columns=list(STATES))
    state = smooth_states(probs, cfg.switch_prob, cfg.switch_days)
    out = probs.copy()
    out["state"] = state
    out["mult"] = state.map(mult).astype(float)
    return out.reindex(feat.index)
