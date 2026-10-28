"""IC-IR factor combination (03 §3.4), point-in-time.

At each month-end rebalance date d, using only IC observations whose h-day forward window
has CLOSED by d (IC dated t is known at t+h → shifted forward by h):

    w_i ∝ max(mean IC_i, 0) / std(IC_i)       over the trailing ``lookback_days``
    (D28 ``sign_aware``: w_i ∝ |mean IC_i| / std(IC_i) with the factor's sign = sign(mean IC_i))
    w   = (1 − s)·w_norm + s·equal            (s = ``shrink_to_equal``)

Weights hold until the next rebalance. Composite = Σ_i w_i · neutral_z_i (renormalised
over factors available for each name).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from quantswing.core.config import CombineCfg


def ic_ir_weights(ic: pd.DataFrame, h: int, cfg: CombineCfg) -> pd.DataFrame:
    """Daily weights (date × factor) from a daily IC panel (date × factor) at horizon ``h``."""
    known = ic.shift(h)  # IC(t) becomes known at t+h
    idx = pd.DatetimeIndex(ic.index)
    month_end = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).transform("max")
    rebal_dates = sorted(set(month_end))
    rows: dict[pd.Timestamp, pd.Series] = {}
    k = ic.shape[1]
    equal = pd.Series(1.0 / k, index=ic.columns)
    for d in rebal_dates:
        win = known.loc[:d].tail(cfg.lookback_days)
        if win.notna().sum().max() < cfg.min_history_days:
            continue
        m, s = win.mean(), win.std()
        if cfg.sign_aware:  # D28: |IC|/σ magnitude, sign from the (past) mean IC
            sign = np.sign(m).replace(0, 1.0).fillna(1.0)
            raw = (m.abs() / s.where(s > 0)).fillna(0.0)
        else:  # spec 03 §3.4: max(IC, 0)/σ
            sign = pd.Series(1.0, index=m.index)
            raw = (m.clip(lower=0) / s.where(s > 0)).fillna(0.0)
        eq_w = equal
        if cfg.min_t > 0:  # keep only factors whose trailing IC is significant (point-in-time)
            n_eff = win.notna().sum() / max(h, 1)
            t = m.abs() / (s.where(s > 0) / np.sqrt(n_eff.where(n_eff > 0)))
            keep = (t >= cfg.min_t).fillna(False).astype(float)
            raw = raw * keep
            eq_w = keep / keep.sum() if keep.sum() > 0 else equal * 0.0
        w = raw / raw.sum() if raw.sum() > 0 else eq_w
        rows[d] = ((1 - cfg.shrink_to_equal) * w + cfg.shrink_to_equal * eq_w) * sign
    if not rows:
        return pd.DataFrame(np.nan, index=ic.index, columns=ic.columns)
    w_df = pd.DataFrame(rows).T.sort_index()
    # a month-end weight applies from the NEXT session
    return w_df.reindex(ic.index).shift(1).ffill()


def composite(neutral: Iterable[tuple[str, pd.DataFrame]], weights: pd.DataFrame) -> pd.DataFrame:
    """Weighted sum of neutral z panels (streamed one at a time); weights renormalised
    over the factors available for each name/date."""
    num: pd.DataFrame | None = None
    den: pd.DataFrame | None = None
    for name, z in neutral:
        w = weights[name]
        wz = z.mul(w, axis=0)
        wm = z.notna().mul(w.abs(), axis=0).where(z.notna())
        num = wz.fillna(0) if num is None else num.add(wz.fillna(0))
        den = wm.fillna(0) if den is None else den.add(wm.fillna(0))
    assert num is not None and den is not None
    out: pd.DataFrame = num / den.where(den > 0)
    return out
