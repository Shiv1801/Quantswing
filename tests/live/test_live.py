"""D40 live paper: book, end-of-day planner (exact backtest replay) and live poll."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quantswing.live.planner as planner
from quantswing.alpha.research import AUX
from quantswing.core.calendar import TradingCalendar
from quantswing.core.clock import IST
from quantswing.core.config import get_config
from quantswing.live.book import LiveBook
from quantswing.live.poller import poll_once
from quantswing.portfolio.swing import SwingState, run_swing
from quantswing.store.panels import FactorStore
from quantswing.store.repositories import DataPaths
from tests.portfolio.test_swing import _setup

CFG = get_config()


class Cal(TradingCalendar):
    """Every weekday is a session."""

    def __init__(self) -> None:
        super().__init__(CFG.settings.market)

    def is_trading_day(self, d: object) -> bool:  # type: ignore[override]
        return d.weekday() < 5  # type: ignore[attr-defined]


def test_book_cash_moves(tmp_path: Path) -> None:
    b = LiveBook(tmp_path / "l.sqlite")
    b.reset(500000)
    assert b.capital(1.0) == 500000 and not b.started
    with pytest.raises(ValueError):
        b.move_cash(1000, "x")
    b.put("state", SwingState(cash=500000, equity=500000).to_json())
    st = b.move_cash(-100000, "withdraw")
    assert st.cash == 400000 and len(b.cash_moves()) == 1
    with pytest.raises(ValueError):
        b.move_cash(-900000, "too much")


def test_eod_day_by_day_matches_backtest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p, s, g = _setup()
    idx = p.close.index
    paths = DataPaths.relocated(tmp_path, CFG)
    FactorStore(paths.factors).write(AUX, "gate", g.astype("float32"))
    book = LiveBook(paths.live_db)
    book.reset(float(CFG.risk.paper_capital_inr))
    monkeypatch.setattr(planner.TradingCalendar, "from_config", classmethod(lambda cls: Cal()))
    ic = pd.Series(CFG.risk.portfolio.edge_ic, index=idx)
    for k in [99, 100, 101, 104, 140, len(idx) - 1]:
        sub = p.__class__(
            **{
                f: (
                    getattr(p, f).loc[: idx[k]]
                    if f in ("open", "high", "low", "close", "volume", "turnover", "bench")
                    else getattr(p, f)
                )
                for f in p.__dataclass_fields__
            }
        )
        monkeypatch.setattr(planner, "load_panels", lambda *a, _s=sub, **k: _s)
        rep = planner.eod_step(paths, CFG, book, s, ic)
        assert rep.status == "ok"
    rep = planner.eod_step(paths, CFG, book, s, ic)
    assert rep.status == "up to date"
    # the live book equals a straight backtest from the account-opening session
    bt = run_swing(p, s, g, CFG, start=idx[99], edge_ic=ic)
    eq = book.equity().set_index("date")["equity"]
    assert np.allclose(eq.to_numpy(), bt.daily["equity"].to_numpy(), atol=1e-4)
    f = book.fills()
    assert len(f) == len(bt.fills)
    acts = book.actions()
    assert (acts["action"] == "BUY").any()
    filled = acts[acts["status"].str.startswith("Filled")]
    assert len(filled) > 0 and (acts["action"] == "PLACE SL-M").any()


def _book_with_plan(tmp_path: Path) -> LiveBook:
    b = LiveBook(tmp_path / "l.sqlite")
    st = SwingState(
        cash=400000,
        equity=500000,
        last_session="2026-09-25",
        positions=[
            {
                "symbol": "NSE:HELD-EQ",
                "trade_id": 1,
                "entry_date": "2026-09-20",
                "entry_px": 100.0,
                "qty0": 100,
                "qty": 100,
                "s10": 0.05,
                "z0": 2.0,
                "risk_inr": 1000.0,
                "sl0": 92.5,
                "target": 115.0,
                "stop": 92.5,
                "stop_kind": "SL",
                "peak": 104.0,
                "partial": False,
                "held": 4,
                "buy_cost": 10.0,
                "sell_value": 0.0,
                "sell_cost": 0.0,
                "dividends": 0.0,
                "partial_date": None,
                "partial_px": None,
                "partial_qty": 0,
                "exit_flag": "",
            },
        ],
    )
    b.put("state", st.to_json())
    det = {
        "z": 2.2,
        "rf": 0.01,
        "equity": 500000,
        "s10": 0.06,
        "sigma": 0.02,
        "cap": 204.0,
        "prev_close": 200.0,
        "sl_mult": 1.5,
        "tgt_mult": 3.0,
        "tgt_frac": 0.5,
        "max_weight": 0.2,
    }
    for sym in ("NSE:NEW-EQ", "NSE:GAP-EQ"):
        b.add_action(
            {
                "session": "2026-09-28",
                "symbol": sym,
                "action": "BUY",
                "qty": 50,
                "reason": "x",
                "key": f"2026-09-28|{sym}|BUY",
                "detail": det,
            }
        )
    return b


def test_poll_buys_skips_and_stops(tmp_path: Path) -> None:
    b = _book_with_plan(tmp_path)
    quotes = pd.DataFrame(
        [
            {
                "symbol": "NSE:NEW-EQ",
                "lp": 201.0,
                "open_price": 202.0,
                "high_price": 203.0,
                "low_price": 199.0,
                "prev_close_price": 200.0,
            },
            {
                "symbol": "NSE:GAP-EQ",
                "lp": 210.0,
                "open_price": 208.0,
                "high_price": 211.0,
                "low_price": 207.0,
                "prev_close_price": 200.0,
            },
            {
                "symbol": "NSE:HELD-EQ",
                "lp": 91.0,
                "open_price": 99.0,
                "high_price": 99.5,
                "low_price": 90.0,
                "prev_close_price": 100.0,
            },
        ]
    )
    now = datetime(2026, 9, 28, 10, 5, tzinfo=IST)
    rep = poll_once(b, CFG, lambda syms: quotes, now=now, cal=Cal())
    assert rep.status == "ok" and len(rep.events) == 3
    a = b.actions("2026-09-28").set_index("key")
    assert a.loc["2026-09-28|NSE:NEW-EQ|BUY", "status"].startswith("Bought")
    q = int(a.loc["2026-09-28|NSE:NEW-EQ|BUY", "qty"])
    assert q == int(0.01 * 500000 / (202 * 1.5 * 0.06))  # risk ÷ (open − SL)
    assert a.loc["2026-09-28|NSE:GAP-EQ|BUY", "status"].startswith("Skipped")
    assert (a["action"] == "PLACE SL-M").sum() == 1
    ev = b.events()
    assert ev["kind"].tolist().count("stop") == 1
    # the live buy is now visible immediately (not just as an event) - Holdings/Trade Log
    # don't have to wait for tonight's end-of-day update to show it
    live_pos = {p["symbol"]: p for p in b.live_positions()}
    assert "NSE:NEW-EQ" in live_pos and live_pos["NSE:NEW-EQ"]["qty"] == q
    assert "NSE:GAP-EQ" not in live_pos  # skipped, never bought
    log = b.quote_log(symbol="NSE:NEW-EQ")
    assert len(log) == 1 and log.iloc[0]["ltp"] == 201.0
    rep2 = poll_once(b, CFG, lambda syms: quotes, now=now, cal=Cal())
    assert rep2.events == []  # no duplicate alerts
    assert len(b.quote_log(symbol="NSE:NEW-EQ")) == 2  # one row per poll


def test_poll_closed_outside_hours(tmp_path: Path) -> None:
    b = _book_with_plan(tmp_path)
    rep = poll_once(
        b, CFG, lambda s: pd.DataFrame(), now=datetime(2026, 9, 28, 16, 30, tzinfo=IST), cal=Cal()
    )
    assert rep.status == "market closed"
    rep = poll_once(
        b, CFG, lambda s: pd.DataFrame(), now=datetime(2026, 9, 27, 11, 0, tzinfo=IST), cal=Cal()
    )
    assert rep.status == "market closed"  # Sunday


def test_quote_log_tracks_pnl_per_poll(tmp_path: Path) -> None:
    b = _book_with_plan(tmp_path)  # has one held position: NSE:HELD-EQ, qty 100, entry 100.0
    held = {
        "symbol": "NSE:HELD-EQ",
        "qty": 100,
        "entry_px": 100.0,
        "stop": 92.5,
        "target": 115.0,
        "partial": False,
    }
    b.save_quotes(
        pd.DataFrame(
            [{"symbol": "NSE:HELD-EQ", "ltp": 105.0, "open": 101, "high": 106, "low": 100, "prev_close": 100}]
        )
    )
    b.log_positions({"NSE:HELD-EQ": 105.0}, [held])
    b.save_quotes(
        pd.DataFrame(
            [{"symbol": "NSE:HELD-EQ", "ltp": 110.0, "open": 101, "high": 111, "low": 100, "prev_close": 100}]
        )
    )
    b.log_positions({"NSE:HELD-EQ": 110.0}, [held])
    log = b.quote_log()
    assert len(log) == 2  # one row per poll, newest first
    assert log.iloc[0]["ltp"] == 110.0 and log.iloc[0]["pnl"] == pytest.approx(1000.0)
    assert log.iloc[1]["ltp"] == 105.0 and log.iloc[1]["pnl"] == pytest.approx(500.0)
    assert log.iloc[0]["pnl_pct"] == pytest.approx(0.10)
    filtered = b.quote_log(symbol="NSE:HELD-EQ")
    assert len(filtered) == 2
    # the latest-quote snapshot table still only keeps one row per symbol
    assert len(b.quotes()) == 1


def test_detail_json_roundtrip(tmp_path: Path) -> None:
    b = _book_with_plan(tmp_path)
    d = json.loads(b.actions("2026-09-28").iloc[0]["detail"])
    assert d["cap"] == 204.0


def test_active_run_and_stop(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    from quantswing.live.proc import kill_tree, pid_alive

    b = LiveBook(tmp_path / "l.sqlite")
    rid = b.run_start("eod", pid=999999)  # a process that does not exist
    assert b.active_run("eod") is None  # closed as 'stopped' automatically
    assert b.runs("eod").iloc[0]["status"] == "stopped"
    rid = b.run_start("eod", pid=os.getpid())
    assert b.active_run("eod")["id"] == rid  # type: ignore[index]
    b.run_end(rid, "ok")
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    assert pid_alive(p.pid)
    assert kill_tree(p.pid) or p.wait(5) is not None
    p.wait(5)
    assert not pid_alive(p.pid)
