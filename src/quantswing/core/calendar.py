"""NSE trading calendar.

Sources, in priority order:
1. **Observed sessions** (P1): ``settings.data.calendar_file`` — every date on which the
   calendar index (``NSE:NIFTY50-INDEX``) has a Fyers daily candle. Inside the observed
   range, a date is a trading day **iff** it is in this set (exact, holidays included).
2. **Rule fallback** outside that range (e.g. future dates): not a configured weekend day
   and not in ``settings.market.holidays``, unless listed in ``special_sessions``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import date, datetime, timedelta
from pathlib import Path

import pyarrow.parquet as pq

from quantswing.core.clock import ist_combine, now_ist
from quantswing.core.config import MarketCfg, get_config


class TradingCalendar:
    """Observed-session calendar with rule fallback; session times in IST."""

    def __init__(self, market: MarketCfg, sessions: Iterable[date] | None = None) -> None:
        self.m = market
        self.holidays = {date.fromisoformat(d) for d in market.holidays}
        self.special = {date.fromisoformat(d) for d in market.special_sessions}
        self.weekend = set(market.weekend_days)
        self.sessions: frozenset[date] = frozenset(sessions or ())
        self.first = min(self.sessions) if self.sessions else None
        self.last = max(self.sessions) if self.sessions else None

    @classmethod
    def from_file(cls, market: MarketCfg, path: Path) -> TradingCalendar:
        """Build with observed sessions from ``path`` (rule-only if the file is absent)."""
        sessions: list[date] = []
        if path.exists():
            sessions = pq.read_table(path, columns=["date"]).column("date").to_pylist()
        return cls(market, sessions)

    @classmethod
    def from_config(cls) -> TradingCalendar:
        """Build from config, loading observed sessions if the calendar file exists."""
        cfg = get_config()
        return cls.from_file(
            cfg.settings.market, cfg.settings.paths.resolve(cfg.settings.data.calendar_file)
        )

    def _observed(self, d: date) -> bool:
        return self.first is not None and self.last is not None and self.first <= d <= self.last

    def is_trading_day(self, d: date) -> bool:
        """True if ``d`` is an NSE trading day."""
        if self._observed(d):
            return d in self.sessions
        if d in self.special:
            return True
        return d.weekday() not in self.weekend and d not in self.holidays

    def next_trading_day(self, d: date) -> date:
        """First trading day strictly after ``d``."""
        n = d + timedelta(days=1)
        while not self.is_trading_day(n):
            n += timedelta(days=1)
        return n

    def prev_trading_day(self, d: date) -> date:
        """Last trading day strictly before ``d``."""
        p = d - timedelta(days=1)
        while not self.is_trading_day(p):
            p -= timedelta(days=1)
        return p

    def trading_days(self, start: date, end: date) -> Iterator[date]:
        """Trading days in the inclusive range [start, end]."""
        d = start
        while d <= end:
            if self.is_trading_day(d):
                yield d
            d += timedelta(days=1)

    def session_bounds(self, d: date) -> tuple[datetime, datetime]:
        """(open, close) IST datetimes for date ``d``."""
        return ist_combine(d, self.m.session_open), ist_combine(d, self.m.session_close)

    def is_market_open(self, at: datetime | None = None) -> bool:
        """True if ``at`` (default: now IST) is inside the regular session of a trading day."""
        at = at or now_ist()
        if not self.is_trading_day(at.date()):
            return False
        o, c = self.session_bounds(at.date())
        return o <= at <= c
