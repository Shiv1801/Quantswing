"""Fyers market-data client: chunked ``history()`` and batched ``quotes()``.

All calls go through the central :class:`~quantswing.broker.throttler.Throttler`.

History
-------
Fyers caps the date span per request by resolution (``settings.fyers.history.
max_days_per_request``: D → 366, intraday → 100, seconds → 30; VERIFY). A request for
[start, end] is split into consecutive inclusive windows of at most ``cap`` calendar days;
results are concatenated, de-duplicated on ``ts`` and sorted.

Fyers returns candles as ``[epoch_utc_s, o, h, l, c, v]``. They are converted to a
tz-aware IST ``ts`` column (08 §3). Daily candles are stamped at 00:00 IST of the session
date, so ``ts.dt.date`` is the trading date.

Quotes
------
Symbols are sent in batches of ``quotes.max_symbols_per_request`` (50) as a
comma-separated list; each batch costs one API call.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from functools import partial
from typing import Any

import pandas as pd

from quantswing.broker.throttler import Throttler
from quantswing.core.clock import epoch_series_to_ist
from quantswing.core.config import AppConfig, get_config
from quantswing.core.logging import get_logger
from quantswing.core.types import (
    CANDLE_COLUMNS,
    DAILY_RESOLUTIONS,
    INTRADAY_RESOLUTIONS,
    SECOND_RESOLUTIONS,
    BrokerError,
)

log = get_logger(__name__)


# --------------------------------------------------------------------------- pure helpers
def max_days_for(resolution: str, caps: dict[str, int]) -> int:
    """Per-request day cap for a Fyers resolution string."""
    if resolution in caps:
        return caps[resolution]
    if resolution in DAILY_RESOLUTIONS:
        return caps["D"]
    if resolution in INTRADAY_RESOLUTIONS:
        return caps["intraday"]
    if resolution in SECOND_RESOLUTIONS:
        return caps["seconds"]
    raise ValueError(f"unknown Fyers resolution: {resolution!r}")


def chunk_date_range(start: date, end: date, max_days: int) -> list[tuple[date, date]]:
    """Split inclusive [start, end] into consecutive inclusive windows of ≤ ``max_days`` days.

    Windows do not overlap: next window starts the day after the previous one ends.
    """
    if max_days < 1:
        raise ValueError("max_days must be ≥ 1")
    if end < start:
        raise ValueError(f"end {end} before start {start}")
    out: list[tuple[date, date]] = []
    s = start
    while s <= end:
        e = min(end, s + timedelta(days=max_days - 1))
        out.append((s, e))
        s = e + timedelta(days=1)
    return out


def candles_to_frame(candles: Sequence[Sequence[float]]) -> pd.DataFrame:
    """Fyers ``candles`` (``[epoch_utc_s, o, h, l, c, v]`` rows) → DataFrame with IST ``ts``.

    Output columns: ``CANDLE_COLUMNS``; ``ts`` is tz-aware Asia/Kolkata; prices float64,
    volume int64; sorted by ``ts``, duplicates dropped (last wins).
    """
    if len(candles) == 0:
        return empty_candles()
    raw = pd.DataFrame(list(candles), columns=list(CANDLE_COLUMNS))
    df = pd.DataFrame(
        {
            "ts": epoch_series_to_ist(raw["ts"]),
            "open": raw["open"].astype("float64"),
            "high": raw["high"].astype("float64"),
            "low": raw["low"].astype("float64"),
            "close": raw["close"].astype("float64"),
            "volume": raw["volume"].astype("float64").round().astype("int64"),
        }
    )
    return df.drop_duplicates(subset="ts", keep="last").sort_values("ts").reset_index(drop=True)


def empty_candles() -> pd.DataFrame:
    """Empty candle frame with the correct dtypes."""
    return pd.DataFrame(
        {
            "ts": pd.Series([], dtype="datetime64[ns, Asia/Kolkata]"),
            "open": pd.Series([], dtype="float64"),
            "high": pd.Series([], dtype="float64"),
            "low": pd.Series([], dtype="float64"),
            "close": pd.Series([], dtype="float64"),
            "volume": pd.Series([], dtype="int64"),
        }
    )


def batched(items: Sequence[str], size: int) -> list[list[str]]:
    """Split ``items`` into lists of at most ``size``."""
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


# --------------------------------------------------------------------------- client
class FyersData:
    """Throttled data access over an authenticated ``fyersModel.FyersModel``."""

    def __init__(self, client: Any, throttler: Throttler, cfg: AppConfig | None = None) -> None:
        self.client = client
        self.throttler = throttler
        self.cfg = cfg or get_config()

    def _check(self, resp: Any, what: str) -> dict[str, Any]:
        fc = self.cfg.settings.fyers
        if not isinstance(resp, dict):
            raise BrokerError(f"{what}: unexpected response type {type(resp).__name__}", resp)
        if resp.get("s") not in (fc.success_status, fc.no_data_status):
            raise BrokerError(f"{what}: {resp.get('message', resp)}", resp)
        return resp

    def history(
        self,
        symbol: str,
        start: date,
        end: date,
        resolution: str = "D",
        *,
        skip_failed_chunks: bool = False,
    ) -> pd.DataFrame:
        """OHLCV candles for ``symbol`` over inclusive [start, end], chunked per Fyers caps.

        Returns a frame with ``CANDLE_COLUMNS`` (IST ``ts``); empty if no data.
        With ``skip_failed_chunks``, a failing window (e.g. a range older than Fyers keeps)
        is logged and skipped as long as at least one window succeeds; if every window
        fails, the last error is raised.
        """
        hc = self.cfg.settings.fyers.history
        windows = chunk_date_range(start, end, max_days_for(resolution, hc.max_days_per_request))
        frames: list[pd.DataFrame] = []
        errors: list[BrokerError] = []
        for s, e in windows:
            req = {
                "symbol": symbol,
                "resolution": resolution,
                "date_format": hc.date_format,
                "range_from": s.isoformat(),
                "range_to": e.isoformat(),
                "cont_flag": hc.cont_flag,
            }
            try:
                resp = self._check(
                    self.throttler.call(partial(self.client.history, data=req), label="history"),
                    f"history {symbol} {s}..{e}",
                )
            except BrokerError as err:
                if not skip_failed_chunks:
                    raise
                errors.append(err)
                log.warning("history window failed", extra={"symbol": symbol, "error": str(err)})
                continue
            frames.append(candles_to_frame(resp.get("candles") or []))
        if errors and len(errors) == len(windows):
            raise errors[-1]
        non_empty = [f for f in frames if not f.empty]
        out = (
            pd.concat(non_empty, ignore_index=True)
            .drop_duplicates(subset="ts", keep="last")
            .sort_values("ts")
            .reset_index(drop=True)
            if non_empty
            else empty_candles()
        )
        log.info(
            "history fetched",
            extra={
                "symbol": symbol,
                "resolution": resolution,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "requests": len(windows),
                "rows": len(out),
            },
        )
        return out

    def quotes(self, symbols: Sequence[str]) -> pd.DataFrame:
        """Snapshot quotes for many symbols, batched (1 API call per batch).

        Returns one row per symbol: ``symbol`` plus the flattened Fyers ``v`` fields
        (``lp``, ``open_price``, ``high_price``, ``low_price``, ``prev_close_price``,
        ``volume``, ``ch``, ``chp``, ...).
        """
        size = self.cfg.settings.fyers.quotes.max_symbols_per_request
        rows: list[dict[str, Any]] = []
        for batch in batched(list(symbols), size):
            req = {"symbols": ",".join(batch)}
            resp = self._check(
                self.throttler.call(partial(self.client.quotes, data=req), label="quotes"),
                f"quotes batch of {len(batch)}",
            )
            for item in resp.get("d") or []:
                v = item.get("v") or {}
                if isinstance(v, dict):
                    rows.append({"symbol": item.get("n", v.get("symbol")), **v})
        return pd.DataFrame(rows)

    def profile(self) -> dict[str, Any]:
        """Account profile (used as a cheap token-validity check)."""
        return self._check(self.throttler.call(self.client.get_profile, label="profile"), "profile")
