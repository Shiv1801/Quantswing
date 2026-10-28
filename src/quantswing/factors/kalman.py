"""KAL — local-linear-trend Kalman filter on ln P (03 §2).

State ``[level, slope]``; ``F = [[1,1],[0,1]]``, ``H = [1, 0]``,
``Q = diag(level_var, slope_var)``, ``R = obs_var``. Filtered (causal) estimates only.
Outputs slope (daily log drift) and its t-stat ``slope / sqrt(P_slope)``.
Missing observations → predict step only. Vectorised across symbols; loop over time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantswing.core.config import KalmanCfg
from quantswing.factors.common import Panels


def kalman_trend(logp: pd.DataFrame, cfg: KalmanCfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filtered slope and slope t-stat for each column of ``logp``."""
    y = logp.to_numpy(dtype="float64")
    t_len, n = y.shape
    lvl = np.full(n, np.nan)
    slp = np.zeros(n)
    p00 = np.zeros(n)
    p01 = np.zeros(n)
    p11 = np.zeros(n)
    started = np.zeros(n, dtype=bool)
    out_s = np.full((t_len, n), np.nan)
    out_t = np.full((t_len, n), np.nan)
    q0, q1, rv = cfg.level_var, cfg.slope_var, cfg.obs_var
    for t in range(t_len):
        obs = y[t]
        has = ~np.isnan(obs)
        new = has & ~started
        lvl[new], slp[new] = obs[new], 0.0
        p00[new], p01[new], p11[new] = rv, 0.0, cfg.init_slope_var
        started |= new
        run = started & ~new
        # predict: x = F x ; P = F P F' + Q
        lvl_p = lvl + slp
        a00 = p00 + 2 * p01 + p11 + q0
        a01 = p01 + p11
        a11 = p11 + q1
        lvl = np.where(run, lvl_p, lvl)
        p00 = np.where(run, a00, p00)
        p01 = np.where(run, a01, p01)
        p11 = np.where(run, a11, p11)
        # update where observed
        upd = run & has
        s = p00 + rv
        k0, k1 = p00 / s, p01 / s
        innov = np.where(upd, obs - lvl, 0.0)
        lvl = np.where(upd, lvl + k0 * innov, lvl)
        slp = np.where(upd, slp + k1 * innov, slp)
        n00 = (1 - k0) * p00
        n01 = (1 - k0) * p01
        n11 = p11 - k1 * p01
        p00 = np.where(upd, n00, p00)
        p01 = np.where(upd, n01, p01)
        p11 = np.where(upd, n11, p11)
        emit = started & has
        out_s[t, emit] = slp[emit]
        out_t[t, emit] = slp[emit] / np.sqrt(p11[emit])
    idx, cols = logp.index, logp.columns
    return pd.DataFrame(out_s, index=idx, columns=cols), pd.DataFrame(
        out_t, index=idx, columns=cols
    )


def compute(p: Panels, cfg: KalmanCfg) -> dict[str, pd.DataFrame]:
    """``{"KAL_SLOPE": ..., "KAL_T": ...}``."""
    s, t = kalman_trend(p.logp, cfg)
    return {"KAL_SLOPE": s, "KAL_T": t}
