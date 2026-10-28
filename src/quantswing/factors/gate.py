"""Universe integrity gate (03 §1), derivable subset under D22.

Pass on date t iff all of:
* traded at t (close present and volume > 0);
* ≥ ``min_history`` sessions with a close up to t;
* traded on ≥ ``min_continuity`` of the last ``continuity_window`` sessions;
* longest circuit-locked streak in the last ``lock_window`` sessions ≤ ``max_lock_streak``,
  where a locked day = ``high == low`` and close changed (one-price day at a limit).

* not under an NSE surveillance measure that day (ASM/GSM/ESM/IRP/default/BZ — D31, from
  the daily indicator file; before that file exists nothing is excluded).

Results-window entry block: :mod:`quantswing.alpha.events` (entries only, not the gate).
Capacity is a sizing constraint (P5).
"""

from __future__ import annotations

import pandas as pd

from quantswing.core.config import GateCfg
from quantswing.factors.common import Panels, run_length


def locked_days(p: Panels) -> pd.DataFrame:
    """One-price days with a price change (proxy for circuit-locked sessions)."""
    changed = p.close.ne(p.close.shift(1)) & p.close.shift(1).notna()
    out: pd.DataFrame = p.high.eq(p.low) & changed & p.close.notna()
    return out


def universe_gate(p: Panels, cfg: GateCfg) -> pd.DataFrame:
    """Boolean panel: True where the symbol passes the gate on that date."""
    has = p.close.notna() & (p.volume.fillna(0) > 0)  # a zero-volume print is not a trade
    history = has.cumsum() >= cfg.min_history
    cont = has.rolling(cfg.continuity_window, min_periods=cfg.continuity_window).mean()
    continuity = cont >= cfg.min_continuity
    streak = run_length(locked_days(p))
    no_lock = streak.rolling(cfg.lock_window, min_periods=1).max() <= cfg.max_lock_streak
    out: pd.DataFrame = has & history & continuity & no_lock
    if cfg.exclude_surveillance and p.restricted is not None:
        out = out & ~p.restricted.reindex_like(out).fillna(False).astype(bool)
    return out
