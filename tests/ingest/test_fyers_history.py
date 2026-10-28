"""Resumable history loader: full, incremental, skip, failures, abort, session end."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from quantswing.core.calendar import TradingCalendar
from quantswing.core.clock import IST
from quantswing.core.config import get_config
from quantswing.core.types import RateLimitExhausted
from quantswing.ingest.fyers_history import HistoryLoader, latest_completed_session
from quantswing.store.repositories import JsonState, PriceStore
from tests.conftest import FakeFD, make_candles, weekdays

DATES = weekdays(date(2024, 1, 1), date(2026, 9, 25))


def loader(tmp_path: Path, fd: FakeFD) -> HistoryLoader:
    return HistoryLoader(
        fd,  # type: ignore[arg-type]
        PriceStore(tmp_path / "eq"),
        JsonState(tmp_path / "state.json"),
        kind="eq",
        cfg=get_config(),
    )


def test_full_then_skip_then_incremental(tmp_path: Path) -> None:
    full = make_candles(DATES)
    fd = FakeFD({"NSE:A-EQ": full[full["ts"].dt.date <= date(2026, 9, 18)]})
    ld = loader(tmp_path, fd)
    s = ld.run(["NSE:A-EQ"], date(2015, 1, 1), date(2026, 9, 18))
    assert s.fetched == 1 and fd.calls[-1][1] == date(2015, 1, 1)
    st = ld.state.get("eq:NSE:A-EQ")
    assert st["complete"] and st["first"] == "2024-01-01" and st["last"] == "2026-09-18"

    s = ld.run(["NSE:A-EQ"], date(2015, 1, 1), date(2026, 9, 18))
    assert s.skipped == 1 and len(fd.calls) == 1  # no API call when up to date

    fd.series["NSE:A-EQ"] = full  # a week of new candles
    s = ld.run(["NSE:A-EQ"], date(2015, 1, 1), date(2026, 9, 25))
    assert fd.calls[-1][1] == date(2026, 9, 8)  # last - 10 day lookback
    stored = ld.store.read("NSE:A-EQ")
    assert stored is not None and len(stored) == len(DATES)
    assert stored["date"].is_monotonic_increasing and stored["date"].is_unique


def test_state_survives_restart(tmp_path: Path) -> None:
    fd = FakeFD({"NSE:A-EQ": make_candles(DATES)})
    loader(tmp_path, fd).run(["NSE:A-EQ"], date(2015, 1, 1), date(2026, 9, 25))
    fd2 = FakeFD({"NSE:A-EQ": make_candles(DATES)})
    s = loader(tmp_path, fd2).run(["NSE:A-EQ"], date(2015, 1, 1), date(2026, 9, 25))
    assert s.skipped == 1 and fd2.calls == []


def test_failures_recorded_and_run_continues(tmp_path: Path) -> None:
    fd = FakeFD({"NSE:B-EQ": make_candles(DATES)}, fail={"NSE:A-EQ"})
    s = loader(tmp_path, fd).run(
        ["NSE:A-EQ", "NSE:B-EQ", "NSE:C-EQ"], date(2015, 1, 1), date(2026, 9, 25)
    )
    assert list(s.failed) == ["NSE:A-EQ"] and s.fetched == 1 and s.empty == 1
    assert s.aborted is None


def test_abort_after_consecutive_failures(tmp_path: Path) -> None:
    n = get_config().settings.data.abort_after_consecutive_failures
    syms = [f"NSE:X{i}-EQ" for i in range(n + 5)]
    s = loader(tmp_path, FakeFD({}, fail=set(syms))).run(syms, date(2015, 1, 1), date(2026, 9, 25))
    assert s.aborted and len(s.failed) == n


def test_daily_cap_stops_run(tmp_path: Path) -> None:
    class Capped(FakeFD):
        def history(self, symbol, start, end, resolution="D", **_):  # type: ignore[no-untyped-def]
            raise RateLimitExhausted("cap")

    s = loader(tmp_path, Capped({})).run(
        ["NSE:A-EQ", "NSE:B-EQ"], date(2015, 1, 1), date(2026, 9, 25)
    )
    assert s.aborted and s.attempted == 1


def test_latest_completed_session() -> None:
    cal = TradingCalendar(get_config().settings.market)
    assert latest_completed_session(cal, datetime(2026, 9, 25, 12, 0, tzinfo=IST)) == date(
        2026, 9, 24
    )
    assert latest_completed_session(cal, datetime(2026, 9, 25, 16, 0, tzinfo=IST)) == date(
        2026, 9, 25
    )
    assert latest_completed_session(cal, datetime(2026, 9, 27, 10, 0, tzinfo=IST)) == date(
        2026, 9, 25
    )
