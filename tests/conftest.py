"""Shared fixtures: synthetic candles, fake Fyers data client, temp data paths."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quantswing.broker.fyers_data import empty_candles
from quantswing.core.clock import IST
from quantswing.core.types import BrokerError
from quantswing.store.repositories import DataPaths


def weekdays(start: date, end: date) -> list[date]:
    """Mon–Fri dates in [start, end]."""
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def make_candles(dates: list[date], start_price: float = 100.0, seed: int = 0) -> pd.DataFrame:
    """Fyers-style candle frame (``CANDLE_COLUMNS``) with a gentle random walk."""
    if not dates:
        return empty_candles()
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, len(dates))
    close = start_price * np.exp(np.cumsum(r))
    open_ = np.r_[start_price, close[:-1]]
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995
    ts = pd.to_datetime([pd.Timestamp(d) for d in dates]).tz_localize(IST)
    return pd.DataFrame(
        {
            "ts": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(1_000, 100_000, len(dates)).astype("int64"),
        }
    )


class FakeFD:
    """Stands in for FyersData.history: serves slices of pre-built series."""

    def __init__(self, series: dict[str, pd.DataFrame], fail: set[str] | None = None) -> None:
        self.series = series
        self.fail = fail or set()
        self.calls: list[tuple[str, date, date]] = []

    def history(
        self, symbol: str, start: date, end: date, resolution: str = "D", **_: Any
    ) -> pd.DataFrame:
        self.calls.append((symbol, start, end))
        if symbol in self.fail:
            raise BrokerError(f"history {symbol}: invalid symbol")
        df = self.series.get(symbol)
        if df is None:
            return make_candles([])
        d = df["ts"].dt.date
        return df[(d >= start) & (d <= end)].reset_index(drop=True)


@pytest.fixture
def tmp_paths(tmp_path: Path) -> DataPaths:
    """DataPaths redirected into a temp dir."""
    return DataPaths.relocated(tmp_path)


def symbol_master_csv(rows: list[dict[str, Any]]) -> str:
    """Build NSE_CM.csv text (21 columns, no header) from dicts."""
    lines = []
    for r in rows:
        cols = [""] * 21
        cols[0] = r.get("fytoken", "101")
        cols[1] = r["name"]
        cols[2] = str(r.get("itype", 0))
        cols[3] = "1"
        cols[4] = "0.05"
        cols[5] = r.get("isin", "INE000A01010")
        cols[9] = r["ticker"]
        cols[10] = "10"
        lines.append(",".join(cols))
    return "\n".join(lines) + "\n"
