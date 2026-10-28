"""Results-window entry block (03 §1 "No new entry within ±3 sessions of results", D31).

Decisions are made at the close of ``t`` for entry at the open of ``s = t + 1``. A new
entry in symbol j is blocked when, using only what is known at the close of ``t``:

* **after results** — results were published at session ``e ≤ t`` and ``s − e ≤ after``;
* **before results** — a results board meeting at session ``m`` was announced (known at
  ``k ≤ t``) and ``0 ≤ m − s ≤ before``.

Returns a boolean panel (True = no new entry). S5 (post-earnings drift) ignores it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantswing.core.config import EventWindowCfg
from quantswing.factors.common import Panels


def _mark(out: np.ndarray, rows_from: np.ndarray, rows_to: np.ndarray, cols: np.ndarray) -> None:
    for a, b, c in zip(rows_from, rows_to, cols, strict=False):
        lo, hi = max(int(a), 0), min(int(b), out.shape[0] - 1)
        if lo <= hi:
            out[lo : hi + 1, c] = True


def event_block(p: Panels, cfg: EventWindowCfg) -> pd.DataFrame | None:
    """Boolean block panel, or ``None`` when disabled / no event data loaded."""
    if not cfg.enabled or (p.events is None and p.meetings is None):
        return None
    idx, cols = p.close.index, p.close.columns
    colpos = pd.Series(np.arange(len(cols)), index=cols)
    out = np.zeros(p.close.shape, dtype=bool)
    n = len(idx)
    if p.events is not None and not p.events.empty:
        ev = p.events[p.events["symbol"].isin(cols)]
        e = idx.searchsorted(pd.DatetimeIndex(ev["event_date"]))
        ok = e < n
        # t ∈ [e, e + after − 1]  (s = t + 1 ≤ e + after)
        _mark(out, e[ok], e[ok] + cfg.after - 1, colpos.loc[ev["symbol"]].to_numpy()[ok])
    if p.meetings is not None and not p.meetings.empty:
        mt = p.meetings[p.meetings["symbol"].isin(cols)]
        m = idx.searchsorted(pd.DatetimeIndex(mt["meeting_session"]))
        k = idx.searchsorted(pd.DatetimeIndex(mt["known_session"]))
        # s ∈ [m − before, m]  ⇔ t ∈ [m − before − 1, m − 1], and t ≥ k (announced)
        lo = np.maximum(m - cfg.before - 1, k)
        _mark(out, lo, np.minimum(m - 1, n - 1), colpos.loc[mt["symbol"]].to_numpy())
    return pd.DataFrame(out, index=idx, columns=cols)
