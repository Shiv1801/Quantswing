"""IST clock helpers. All timestamps in QuantSwing are tz-aware Asia/Kolkata (08 §3).

Kept free of config imports so it can be used everywhere (including logging).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

IST_NAME = "Asia/Kolkata"
IST = ZoneInfo(IST_NAME)


def now_ist() -> datetime:
    """Current wall-clock time, tz-aware IST."""
    return datetime.now(tz=IST)


def today_ist() -> date:
    """Current IST calendar date."""
    return now_ist().date()


def to_ist(dt: datetime) -> datetime:
    """Convert a datetime to IST. Naive datetimes are rejected (ambiguous)."""
    if dt.tzinfo is None:
        raise ValueError("naive datetime; attach a timezone first")
    return dt.astimezone(IST)


def epoch_to_ist(epoch_s: int | float) -> datetime:
    """Convert a UTC epoch (seconds) to a tz-aware IST datetime."""
    return datetime.fromtimestamp(float(epoch_s), tz=UTC).astimezone(IST)


def epoch_series_to_ist(epochs: pd.Series) -> pd.Series:
    """Vectorised UTC-epoch-seconds → tz-aware IST ``datetime64`` series."""
    return pd.to_datetime(epochs.astype("int64"), unit="s", utc=True).dt.tz_convert(IST_NAME)


def ist_combine(d: date, t: time) -> datetime:
    """Build a tz-aware IST datetime from a date and a wall-clock time."""
    return datetime.combine(d, t, tzinfo=IST)
