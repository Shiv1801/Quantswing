"""Resumable Fyers daily-history loader (backfill + incremental), per symbol.

State (``backfill_state.json``) per key ``"<kind>:<symbol>"``::

    {"complete": true, "first": "2016-01-01", "last": "2026-09-25", "rows": 2400,
     "error": null, "updated": "..."}

Rules
-----
* Not complete → full fetch ``[history_start, end]`` (chunked by the Fyers per-request cap).
* Complete → incremental fetch ``[last − incremental_lookback_days, end]`` and upsert (the
  overlap re-reads recent candles in case Fyers revised them).
* Skip when complete and ``last ≥ end``.
* ``end`` = latest *completed* session: today after the session close, else the previous
  trading day — so a partial intraday candle is never stored.
* Per-symbol broker errors are recorded and the run continues; after
  ``abort_after_consecutive_failures`` consecutive failures (e.g. token expired) the run
  stops. A daily-cap hit (:class:`RateLimitExhausted`) stops immediately. Re-running
  resumes where it left off.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from quantswing.broker.fyers_data import FyersData
from quantswing.core.calendar import TradingCalendar
from quantswing.core.clock import now_ist
from quantswing.core.config import AppConfig, get_config
from quantswing.core.logging import get_logger
from quantswing.core.types import BrokerError, RateLimitExhausted
from quantswing.store.repositories import JsonState, PriceStore, candles_for_store

log = get_logger(__name__)


def latest_completed_session(cal: TradingCalendar, now: datetime | None = None) -> date:
    """Last session whose daily candle is final at ``now`` (IST)."""
    now = now or now_ist()
    d = now.date()
    if cal.is_trading_day(d) and now.time() >= cal.m.session_close:
        return d
    return cal.prev_trading_day(d)


@dataclass
class RunSummary:
    """Outcome of a loader run."""

    attempted: int = 0
    fetched: int = 0
    skipped: int = 0
    empty: int = 0
    failed: dict[str, str] = field(default_factory=dict)
    aborted: str | None = None
    rows_written: int = 0


class HistoryLoader:
    """Fetches daily candles for many symbols into a :class:`PriceStore`."""

    def __init__(
        self,
        fd: FyersData,
        store: PriceStore,
        state: JsonState,
        *,
        kind: str,
        cfg: AppConfig | None = None,
    ) -> None:
        self.fd = fd
        self.store = store
        self.state = state
        self.kind = kind
        self.cfg = cfg or get_config()

    def key(self, symbol: str) -> str:
        """State key for ``symbol``."""
        return f"{self.kind}:{symbol}"

    def plan(self, symbol: str, start: date, end: date) -> tuple[date, date] | None:
        """Date window to fetch for ``symbol``, or ``None`` if up to date."""
        st = self.state.get(self.key(symbol)) or {}
        if st.get("complete") and st.get("last"):
            last = date.fromisoformat(st["last"])
            if last >= end:
                return None
            lb = self.cfg.settings.data.incremental_lookback_days
            return max(start, last - timedelta(days=lb)), end
        return start, end

    def load_symbol(self, symbol: str, start: date, end: date) -> int | None:
        """Fetch + upsert one symbol. Returns rows stored (``None`` if skipped)."""
        window = self.plan(symbol, start, end)
        if window is None:
            return None
        s, e = window
        candles = self.fd.history(
            symbol, s, e, self.cfg.settings.data.resolution, skip_failed_chunks=True
        )
        prev = self.state.get(self.key(symbol)) or {}
        if candles.empty:
            merged_rows = int(prev.get("rows", 0))
            first, last = prev.get("first"), prev.get("last")
        else:
            merged = self.store.upsert(symbol, candles_for_store(candles, symbol, now_ist()))
            merged_rows = len(merged)
            first, last = merged["date"].iloc[0].isoformat(), merged["date"].iloc[-1].isoformat()
        self.state.set(
            self.key(symbol),
            {
                "complete": True,
                "first": first,
                "last": last,
                "rows": merged_rows,
                "error": None,
                "updated": now_ist().isoformat(timespec="seconds"),
            },
        )
        return 0 if candles.empty else len(candles)

    def run(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        progress: Callable[[int, int, RunSummary], None] | None = None,
    ) -> RunSummary:
        """Load all ``symbols``; resilient to per-symbol errors (see module docstring)."""
        dc = self.cfg.settings.data
        summ = RunSummary()
        consecutive = 0
        for i, sym in enumerate(symbols, 1):
            summ.attempted += 1
            try:
                n = self.load_symbol(sym, start, end)
            except RateLimitExhausted as e:
                summ.aborted = f"daily API cap: {e}"
                break
            except BrokerError as e:
                consecutive += 1
                summ.failed[sym] = str(e)
                prev = self.state.get(self.key(sym)) or {}
                self.state.set(self.key(sym), {**prev, "error": str(e)[:300]})
                log.warning("history failed", extra={"symbol": sym, "error": str(e)[:300]})
                if consecutive >= dc.abort_after_consecutive_failures:
                    summ.aborted = (
                        f"{consecutive} consecutive failures (last: {e}). "
                        "Token expired? Run fyers_login.py, then re-run to resume."
                    )
                    break
            else:
                consecutive = 0
                if n is None:
                    summ.skipped += 1
                elif n == 0:
                    summ.empty += 1
                else:
                    summ.fetched += 1
                    summ.rows_written += n
            if progress and (i % dc.progress_every == 0 or i == len(symbols)):
                progress(i, len(symbols), summ)
        return summ
