"""Validation checks and the adjustment probe."""

from __future__ import annotations

from datetime import date

import pandas as pd

from quantswing.core.calendar import TradingCalendar
from quantswing.core.config import KnownEventCfg, get_config
from quantswing.store.repositories import candles_for_store
from quantswing.store.validation import detect_adjustments, probe_known_event, validate_symbol
from tests.conftest import make_candles, weekdays

VC = get_config().settings.validation
DATES = weekdays(date(2025, 1, 1), date(2025, 12, 31))
CAL = TradingCalendar(get_config().settings.market, DATES)
LATEST = DATES[-1]


def stored(df: pd.DataFrame) -> pd.DataFrame:
    return candles_for_store(df, "NSE:T-EQ", pd.Timestamp("2026-01-01", tz="Asia/Kolkata"))


def checks(issues: list[dict]) -> dict[str, int]:
    return pd.Series([i["check"] for i in issues]).value_counts().to_dict() if issues else {}


def test_clean_series_has_no_issues() -> None:
    assert validate_symbol(stored(make_candles(DATES)), "NSE:T-EQ", VC, CAL, LATEST) == []


def test_ohlc_reject_and_volume_warn() -> None:
    df = make_candles(DATES)
    df.loc[10, "low"] = df.loc[10, "high"] + 1  # L > H
    df.loc[20, "close"] = -1.0
    df.loc[30, "volume"] = 0
    iss = validate_symbol(stored(df), "NSE:T-EQ", VC, CAL, LATEST)
    c = checks(iss)
    assert c["ohlc"] == 2 and c["volume"] == 1
    assert all(i["severity"] == "reject" for i in iss if i["check"] == "ohlc")


def split_series(k: int = 150, factor: float = 5.0) -> pd.DataFrame:
    df = make_candles(DATES)
    for col in ("open", "high", "low", "close"):
        df.loc[k:, col] = df.loc[k:, col] / factor  # split left unadjusted
    return df


def test_unadjusted_split_auto_adjusted_not_spike() -> None:
    iss = validate_symbol(stored(split_series()), "NSE:T-EQ", VC, CAL, LATEST)
    ca = [i for i in iss if i["check"] == "ca_adjusted"]
    assert len(ca) == 1 and ca[0]["date"] == DATES[150] and ca[0]["severity"] == "info"
    assert "spike" not in checks(iss) and "ca_suspect" not in checks(iss)


def test_split_quarantined_when_auto_adjust_off() -> None:
    vc = VC.model_copy(update={"ca_auto_adjust": False})
    iss = validate_symbol(stored(split_series()), "NSE:T-EQ", vc, CAL, LATEST)
    assert checks(iss).get("ca_suspect") == 1


def test_detect_adjustments_matches_ratio() -> None:
    ev = detect_adjustments(stored(split_series(k=120, factor=2.0)), "NSE:T-EQ", VC)
    assert len(ev) == 1 and ev[0]["ex_date"] == DATES[120] and ev[0]["ratio"] == 0.5
    assert detect_adjustments(stored(make_candles(DATES)), "NSE:T-EQ", VC) == []


def test_spike_after_split_still_detected() -> None:
    df = split_series(k=100)
    df.loc[130, "close"] = df.loc[130, "close"] * 1.15
    df.loc[130, "high"] = df.loc[130, "close"] * 1.001
    c = checks(validate_symbol(stored(df), "NSE:T-EQ", VC, CAL, LATEST))
    assert c.get("spike") == 1  # split return excluded from sigma


def test_spike_quarantined() -> None:
    df = make_candles(DATES)
    df.loc[100, "close"] = df.loc[100, "close"] * 1.15  # +15% in a 1%-vol series, no gap open
    df.loc[100, "high"] = df.loc[100, "close"] * 1.001
    iss = validate_symbol(stored(df), "NSE:T-EQ", VC, CAL, LATEST)
    sp = [i for i in iss if i["check"] == "spike"]
    assert sp and sp[0]["date"] == DATES[100] and sp[0]["severity"] == "quarantine"


def test_missing_sessions_and_stale() -> None:
    df = make_candles(DATES[:200]).drop(index=[50, 51, 52]).reset_index(drop=True)
    c = checks(validate_symbol(stored(df), "NSE:T-EQ", VC, CAL, LATEST))
    assert c["missing_sessions"] == 1 and c["stale"] == 1


def test_off_calendar_candle() -> None:
    cal = TradingCalendar(get_config().settings.market, [d for d in DATES if d != DATES[40]])
    c = checks(validate_symbol(stored(make_candles(DATES)), "NSE:T-EQ", VC, cal, LATEST))
    assert c["off_calendar"] == 1


def test_probe_known_event() -> None:
    ev = KnownEventCfg(symbol="NSE:T-EQ", ex_date=DATES[150], ratio=0.5, note="")
    base = make_candles(DATES)
    assert probe_known_event(stored(base), ev, 0.12).verdict == "ADJUSTED"
    un = base.copy()
    for col in ("open", "high", "low", "close"):
        un.loc[150:, col] = un.loc[150:, col] / 2
    p = probe_known_event(stored(un), ev, 0.12)
    assert p.verdict == "UNADJUSTED" and abs(p.observed_ratio - 0.5) < 0.05  # type: ignore[operator]
    assert probe_known_event(None, ev, 0.12).verdict == "NO_DATA"
