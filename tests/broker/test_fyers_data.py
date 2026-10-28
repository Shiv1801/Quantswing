"""Data: candle conversion UTC→IST, chunking, history/quotes against a fake client."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from typing import Any

import pandas as pd
import pytest

from quantswing.broker.fyers_data import (
    FyersData,
    batched,
    candles_to_frame,
    chunk_date_range,
    max_days_for,
)
from quantswing.broker.throttler import Throttler
from quantswing.core.config import get_config
from quantswing.core.types import BrokerError

CAPS = {"D": 366, "1D": 366, "intraday": 100, "seconds": 30}


def epoch(dt: datetime) -> int:
    return int(dt.timestamp())


def test_daily_candle_ist_midnight_maps_to_session_date() -> None:
    # Fyers daily candle for 2026-09-24 is stamped 00:00 IST = 2026-09-23 18:30 UTC.
    ts = epoch(datetime(2026, 9, 23, 18, 30, tzinfo=UTC))
    df = candles_to_frame([[ts, 100, 110, 95, 105, 12345]])
    assert str(df["ts"].dt.tz) == "Asia/Kolkata"
    assert df["ts"].iloc[0].date() == date(2026, 9, 24)
    assert df["ts"].iloc[0].hour == 0


def test_intraday_candle_utc_to_ist_offset() -> None:
    ts = epoch(datetime(2026, 9, 24, 3, 45, tzinfo=UTC))  # 09:15 IST
    df = candles_to_frame([[ts, 1, 1, 1, 1, 1]])
    t = df["ts"].iloc[0]
    assert (t.hour, t.minute) == (9, 15)
    assert t.utcoffset() == timedelta(hours=5, minutes=30)


def test_candles_sorted_deduped_and_typed() -> None:
    a = epoch(datetime(2026, 9, 22, 18, 30, tzinfo=UTC))
    b = epoch(datetime(2026, 9, 23, 18, 30, tzinfo=UTC))
    df = candles_to_frame([[b, 2, 2, 2, 2, 2.0], [a, 1, 1, 1, 1, 1], [b, 3, 3, 3, 3, 3]])
    assert list(df.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert len(df) == 2 and df["ts"].is_monotonic_increasing
    assert df["close"].iloc[-1] == 3.0  # last duplicate wins
    assert df["volume"].dtype == "int64" and df["open"].dtype == "float64"


def test_empty_candles() -> None:
    df = candles_to_frame([])
    assert df.empty and list(df.columns) == ["ts", "open", "high", "low", "close", "volume"]


def test_chunk_date_range_covers_exactly_without_overlap() -> None:
    s, e = date(2024, 1, 1), date(2026, 9, 25)
    chunks = chunk_date_range(s, e, 366)
    assert chunks[0][0] == s and chunks[-1][1] == e
    for (_a0, a1), (b0, _b1) in pairwise(chunks):
        assert b0 == a1 + timedelta(days=1)
    assert all((c1 - c0).days + 1 <= 366 for c0, c1 in chunks)
    assert chunk_date_range(s, s, 100) == [(s, s)]
    with pytest.raises(ValueError):
        chunk_date_range(e, s, 10)


def test_max_days_for() -> None:
    assert max_days_for("D", CAPS) == 366
    assert max_days_for("5", CAPS) == 100
    assert max_days_for("15S", CAPS) == 30
    with pytest.raises(ValueError):
        max_days_for("7", CAPS)


def test_batched() -> None:
    assert batched([str(i) for i in range(120)], 50)[-1] == [str(i) for i in range(100, 120)]


class FakeClient:
    def __init__(self) -> None:
        self.history_calls: list[dict[str, Any]] = []
        self.quote_calls: list[str] = []

    def history(self, data: dict[str, Any]) -> dict[str, Any]:
        self.history_calls.append(data)
        d = date.fromisoformat(data["range_from"])
        ts = epoch(datetime(d.year, d.month, d.day, tzinfo=UTC) - timedelta(hours=5, minutes=30))
        return {"s": "ok", "candles": [[ts, 1, 2, 0.5, 1.5, 10]]}

    def quotes(self, data: dict[str, Any]) -> dict[str, Any]:
        self.quote_calls.append(data["symbols"])
        return {"s": "ok", "d": [{"n": s, "v": {"lp": 1.0}} for s in data["symbols"].split(",")]}


def fd(client: Any) -> FyersData:
    cfg = get_config()
    return FyersData(client, Throttler(cfg.settings.throttle, sleep=lambda s: None), cfg)


def test_history_chunks_requests() -> None:
    c = FakeClient()
    df = fd(c).history("NSE:RELIANCE-EQ", date(2024, 1, 1), date(2026, 9, 25))
    assert len(c.history_calls) == 3  # 999 days / 366
    assert c.history_calls[0]["date_format"] == "1" and c.history_calls[0]["cont_flag"] == "1"
    assert len(df) == 3 and isinstance(df["ts"].dtype, pd.DatetimeTZDtype)
    assert df["ts"].iloc[0].date() == date(2024, 1, 1)


def test_history_raises_on_error() -> None:
    class Bad(FakeClient):
        def history(self, data: dict[str, Any]) -> dict[str, Any]:
            return {"s": "error", "code": -300, "message": "invalid symbol"}

    with pytest.raises(BrokerError, match="invalid symbol"):
        fd(Bad()).history("NSE:XXX-EQ", date(2026, 1, 1), date(2026, 2, 1))


def test_history_no_data_returns_empty() -> None:
    class NoData(FakeClient):
        def history(self, data: dict[str, Any]) -> dict[str, Any]:
            return {"s": "no_data", "candles": []}

    assert fd(NoData()).history("NSE:X-EQ", date(2026, 1, 1), date(2026, 2, 1)).empty


def test_quotes_batched_by_50() -> None:
    c = FakeClient()
    syms = [f"NSE:S{i}-EQ" for i in range(120)]
    q = fd(c).quotes(syms)
    assert len(c.quote_calls) == 3 and len(q) == 120 and "lp" in q.columns


def test_history_skips_failed_old_windows() -> None:
    class OldFails(FakeClient):
        def history(self, data: dict[str, Any]) -> dict[str, Any]:
            if data["range_from"] < "2025-01-01":
                return {"s": "error", "code": -1, "message": "range not supported"}
            return super().history(data)

    c = OldFails()
    df = fd(c).history("NSE:X-EQ", date(2023, 1, 1), date(2026, 9, 25), skip_failed_chunks=True)
    assert len(c.history_calls) == 2 and len(df) == 2  # 2 old windows failed, 2 served
    with pytest.raises(BrokerError):
        fd(OldFails()).history("NSE:X-EQ", date(2023, 1, 1), date(2026, 9, 25))


def test_history_all_windows_fail_raises_even_when_skipping() -> None:
    class Bad(FakeClient):
        def history(self, data: dict[str, Any]) -> dict[str, Any]:
            return {"s": "error", "code": -300, "message": "invalid symbol"}

    with pytest.raises(BrokerError, match="invalid symbol"):
        fd(Bad()).history("NSE:X-EQ", date(2023, 1, 1), date(2026, 9, 25), skip_failed_chunks=True)
