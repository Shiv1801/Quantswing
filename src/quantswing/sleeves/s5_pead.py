"""S5 — post-earnings announcement drift (03 §4 S5, D31: NSE results calendar).

Event session ``e`` = first session at whose close the results were public
(:mod:`quantswing.nse.build`). Setup at ``e``: abnormal return ``r − β·r_b`` ≥
``ar_sigma_min`` × idiosyncratic σ (β, σ from the day before), VOLZ ≥ ``volz_min``, up day.
Confirmation at ``e+1`` ("gap holds on day 1"): the day-1 close keeps ≥ ``gap_hold_frac`` of
the event-day move. Signal at the close of ``e+1`` → entry at the next open (day 2).
Long only (D11). Exempt from the results-window entry block.
"""

from __future__ import annotations

import pandas as pd

from quantswing.core.config import S5Cfg
from quantswing.factors.common import Panels
from quantswing.sleeves.base import FactorView, SleeveSignals


def event_panel(p: Panels) -> pd.DataFrame:
    """True on each (event session, symbol)."""
    out = pd.DataFrame(False, index=p.close.index, columns=p.close.columns)
    if p.events is None or p.events.empty:
        return out
    ev = p.events[p.events["symbol"].isin(out.columns) & p.events["event_date"].isin(out.index)]
    rows = out.index.get_indexer(pd.DatetimeIndex(ev["event_date"]))
    cols = out.columns.get_indexer(pd.Index(ev["symbol"]))
    v = out.to_numpy(copy=True)
    v[rows, cols] = True
    return pd.DataFrame(v, index=out.index, columns=out.columns)


def build(p: Panels, f: FactorView, gate: pd.DataFrame, c: S5Cfg) -> SleeveSignals:
    """S5 signals."""
    ev = event_panel(p)
    r, rb = p.r, p.rb
    ar = r.sub(f["BETA"].shift(1).mul(rb, axis=0))
    z = ar / f["IVOL"].shift(1).where(f["IVOL"].shift(1) > 0)
    move = p.close - p.close.shift(1)
    setup = ev & (z >= c.ar_sigma_min) & (f["VOLZ"] >= c.volz_min) & (move > 0)
    base = p.close.shift(2)
    held = (p.close - base) >= c.gap_hold_frac * (p.close.shift(1) - base)
    sig = setup.shift(1, fill_value=False).astype(bool) & held & gate.astype(bool)
    return SleeveSignals("S5", sig.fillna(False).astype(bool))
