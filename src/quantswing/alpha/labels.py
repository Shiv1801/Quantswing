"""Triple-barrier labels (05 §1) and uniqueness weights.

For an event (symbol, date t) with entry at the close of t:
* risk unit ``R = k_sl · σ_60(t) · P_t``
* upper barrier ``P_t + upper_r · R``, lower ``P_t − R``, vertical ``t + max_hold`` sessions
* scan bars t+1 … t+max_hold: first bar whose HIGH ≥ upper → +1, LOW ≤ lower → −1;
  both on the same bar → −1 (worst case); none → 0 with the vertical-barrier return.
Output per event: ``label``, ``ret`` (log, to the exit), ``t_end`` (exit session),
``holding`` (sessions).

Uniqueness (López de Prado): per symbol, ``c_s`` = number of labels whose span covers
session s; ``w = mean_{s ∈ span} 1/c_s``. Used to down-weight overlapping labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier(
    events: pd.DataFrame,
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    sigma: pd.DataFrame,
    upper_r: float,
    k_sl: float,
    max_hold: int,
) -> pd.DataFrame:
    """Label ``events`` (columns ``date``, ``symbol``) against wide price panels.

    Vectorised over events: loops over the ``max_hold`` look-ahead offsets only.
    """
    idx = close.index
    pos = pd.Series(np.arange(len(idx)), index=idx)
    col = pd.Series(np.arange(close.shape[1]), index=close.columns)
    ev = events[events["symbol"].isin(col.index) & events["date"].isin(idx)].copy()
    r_i = pos.loc[ev["date"]].to_numpy()
    c_i = col.loc[ev["symbol"]].to_numpy()
    cl, hi, lw = close.to_numpy(), high.to_numpy(), low.to_numpy()
    p0 = cl[r_i, c_i]
    unit = k_sl * sigma.to_numpy()[r_i, c_i] * p0
    up, dn = p0 + upper_r * unit, p0 - unit
    n = len(ev)
    label = np.zeros(n, dtype=int)
    end = np.full(n, -1)
    exit_px = np.full(n, np.nan)
    done = ~np.isfinite(unit) | (unit <= 0) | ~np.isfinite(p0)
    last_px = p0.copy()
    last_i = r_i.copy()
    t_len = len(idx)
    for k in range(1, max_hold + 1):
        rows = r_i + k
        valid = (rows < t_len) & ~done
        rr = np.where(rows < t_len, rows, t_len - 1)
        h_k, l_k, c_k = hi[rr, c_i], lw[rr, c_i], cl[rr, c_i]
        traded = valid & np.isfinite(c_k)
        hit_dn = traded & (l_k <= dn)
        hit_up = traded & (h_k >= up) & ~hit_dn
        label[hit_dn], label[hit_up] = -1, 1
        exit_px[hit_dn], exit_px[hit_up] = dn[hit_dn], up[hit_up]
        end[hit_dn | hit_up] = rr[hit_dn | hit_up]
        done |= hit_dn | hit_up
        last_px = np.where(traded, c_k, last_px)
        last_i = np.where(traded, rr, last_i)
    vert = ~done & np.isfinite(unit) & (unit > 0)
    exit_px[vert] = last_px[vert]
    end[vert] = last_i[vert]
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.log(exit_px / p0)
    out = ev[["date", "symbol"]].reset_index(drop=True)
    out["label"] = np.where(vert, np.sign(np.nan_to_num(ret)).astype(int), label)
    out["ret"] = ret
    out["t_end"] = [idx[e] if e >= 0 else pd.NaT for e in end]
    out["holding"] = np.where(end >= 0, end - r_i, 0)
    out["unit"] = unit
    out["p0"] = p0
    res: pd.DataFrame = out[np.isfinite(out["ret"])].reset_index(drop=True)
    return res


def uniqueness_weights(labels: pd.DataFrame, sessions: pd.Index) -> pd.Series:
    """Average uniqueness of each label over its [date, t_end] span, per symbol."""
    pos = pd.Series(np.arange(len(sessions)), index=sessions)
    w = pd.Series(np.nan, index=labels.index)
    for _sym, g in labels.groupby("symbol"):
        s = pos.loc[g["date"]].to_numpy()
        e = pos.loc[g["t_end"]].to_numpy()
        conc = np.zeros(len(sessions) + 1)
        np.add.at(conc, s, 1)
        np.add.at(conc, e + 1, -1)
        c = np.cumsum(conc)[: len(sessions)]
        inv = np.where(c > 0, 1.0 / np.where(c > 0, c, 1), 0.0)
        csum = np.concatenate([[0.0], np.cumsum(inv)])
        w.loc[g.index] = (csum[e + 1] - csum[s]) / (e - s + 1)
    return w
