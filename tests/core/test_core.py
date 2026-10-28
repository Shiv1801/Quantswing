"""Core: config loads + locked decisions, secret masking, calendar stub, clock."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime

import pytest

from quantswing.core.calendar import TradingCalendar
from quantswing.core.clock import IST, epoch_to_ist, to_ist
from quantswing.core.config import MarketCfg, get_config
from quantswing.core.logging import JsonFormatter, SecretMasker


def test_config_loads_and_reflects_locked_decisions() -> None:
    c = get_config()
    assert c.risk.paper_capital_inr == 1_000_000  # D09
    assert c.risk.target_portfolio_vol == 0.15  # D10
    assert c.risk.direction == "long_only" and c.risk.beta_hedge_enabled is False  # D11
    assert (c.risk.risk_per_trade, c.risk.max_heat) == (0.01, 0.06)  # D12
    assert c.sleeves.enabled == {"S1": True, "S2": True, "S3": True, "S4": True, "S5": True}  # D31
    assert c.settings.platform.dashboard == "streamlit"  # D13
    assert c.settings.platform.host == "windows_pc"  # D14
    assert c.settings.platform.alerts == "telegram"  # D17


def test_masker_masks_keys_literals_and_query_params() -> None:
    m = SecretMasker(["access_token", "appIdHash", "auth_code"], ["SUPERSECRET"], 4)
    s = m.mask_str("url?auth_code=abcdefgh&state=x secret=SUPERSECRET")
    assert "abcdefgh" not in s and "SUPERSECRET" not in s and "state=x" in s
    j = m.mask_str('{"access_token": "tok-123456", "s": "ok"}')
    assert "tok-123456" not in j and '"s": "ok"' in j
    d = m.mask({"access_token": "tok-123456", "nested": {"appIdHash": "deadbeef"}, "rows": 5})
    assert d == {"access_token": "tok-****", "nested": {"appIdHash": "dead****"}, "rows": 5}


def test_json_formatter_masks_extras() -> None:
    fmt = JsonFormatter(SecretMasker(["access_token"], ["SUPERSECRET"], 2), "model-v0")
    rec = logging.LogRecord("t", logging.INFO, "f", 1, "hello SUPERSECRET", None, None)
    rec.access_token = "abcdef"
    rec.symbol = "NSE:RELIANCE-EQ"
    out = json.loads(fmt.format(rec))
    assert out["msg"] == "hello SU****"
    assert out["access_token"] == "ab****" and out["symbol"] == "NSE:RELIANCE-EQ"
    assert out["model_version"] == "model-v0" and out["ts"].endswith("+05:30")


def mcfg(**kw: object) -> MarketCfg:
    base = get_config().settings.market.model_dump()
    base.update(kw)
    return MarketCfg(**base)


def test_calendar_stub() -> None:
    cal = TradingCalendar(mcfg(holidays=["2026-10-02"], special_sessions=["2026-11-08"]))
    assert not cal.is_trading_day(date(2026, 9, 26))  # Saturday
    assert not cal.is_trading_day(date(2026, 10, 2))  # configured holiday
    assert cal.is_trading_day(date(2026, 11, 8))  # special Sunday session
    assert cal.next_trading_day(date(2026, 10, 1)) == date(2026, 10, 5)
    assert cal.prev_trading_day(date(2026, 10, 5)) == date(2026, 10, 1)
    assert len(list(cal.trading_days(date(2026, 9, 21), date(2026, 9, 27)))) == 5
    assert cal.is_market_open(datetime(2026, 9, 25, 10, 0, tzinfo=IST))
    assert not cal.is_market_open(datetime(2026, 9, 25, 15, 31, tzinfo=IST))


def test_clock() -> None:
    assert epoch_to_ist(0).isoformat() == "1970-01-01T05:30:00+05:30"
    with pytest.raises(ValueError):
        to_ist(datetime(2026, 1, 1))
