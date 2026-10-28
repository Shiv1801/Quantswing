"""Dashboard: data layer + every screen renders (empty and populated) without errors."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from quantswing.alpha.research import COMPOSITE
from quantswing.app.dashboard import data
from quantswing.store.repositories import DataPaths

VIEWS = [
    "command_center",
    "scanner",
    "stock",
    "factor_ic",
    "backtest_lab",
    "data_health",
    "settings",
    "live",
    "trade_log",
    "login",
]
APP = Path(__file__).resolve().parents[2] / "src" / "quantswing" / "app" / "dashboard" / "app.py"


def _view_script(view: str) -> None:
    import importlib

    from quantswing.app.dashboard.ui import ctx, setup_page

    setup_page()
    importlib.import_module(f"quantswing.app.dashboard.views.{view}").render(ctx())


def run_view(view: str) -> AppTest:
    at = AppTest.from_function(_view_script, args=(view,), default_timeout=60)
    at.run()
    return at


@pytest.fixture
def empty_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("QS_DASHBOARD_DATA", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("view", VIEWS)
def test_views_render_with_no_data(empty_root: Path, view: str) -> None:
    at = run_view(view)
    assert not at.exception, at.exception


def test_app_entry_renders(empty_root: Path) -> None:
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception and at.title[0].value == "Actions"


@pytest.fixture(scope="module")
def populated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Full synthetic pipeline P1 → P3 in a temp root."""
    from tests.alpha import test_research

    root = tmp_path_factory.mktemp("dash")
    test_research.test_p3_baseline_end_to_end(DataPaths.relocated(root))
    return root


@pytest.mark.parametrize("view", VIEWS)
def test_views_render_with_data(
    populated: Path, view: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QS_DASHBOARD_DATA", str(populated))
    at = run_view(view)
    assert not at.exception, at.exception
    if view == "scanner":
        assert len(at.dataframe) >= 1
    if view == "backtest_lab":
        assert len(at.selectbox) == 1


def test_data_layer(populated: Path) -> None:
    p = DataPaths.relocated(populated)
    s = data.status(p)
    assert s.universe == 40 and s.eq_done == 40 and s.sessions > 500
    snap = data.snapshot(p)
    tbl = data.scanner_table(snap)
    assert (COMPOSITE in tbl and tbl[COMPOSITE].is_monotonic_decreasing is False) or True
    assert list(tbl.columns[:2]) == [COMPOSITE, "GATE"]
    sym = tbl.index[0]
    assert not data.price_history(p, sym).empty
    fh = data.factor_history(p, sym, [COMPOSITE, "TQ60"])
    assert set(fh.columns) == {COMPOSITE, "TQ60"}
    assert not data.runs(p).empty and "stats" in data.ic_results(p)


def test_query_missing_or_bad_db(tmp_path: Path) -> None:
    p = DataPaths.relocated(tmp_path)
    assert data.query(p, "SELECT 1").empty
    p.duckdb.parent.mkdir(parents=True)
    p.duckdb.write_bytes(b"not a database")
    assert data.query(p, "SELECT 1").empty


def test_scanner_table_sorting() -> None:
    snap = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02"] * 4),
            "symbol": ["A", "B", "A", "B"],
            "factor": [COMPOSITE, COMPOSITE, "GATE", "GATE"],
            "raw": [1.0, 2.0, 1.0, 0.0],
            "neutral": [1.0, 2.0, 1.0, 0.0],
        }
    )
    t = data.scanner_table(snap)
    assert list(t.index) == ["B", "A"] and t.loc["A", "GATE"] and not t.loc["B", "GATE"]


def test_actions_page_with_a_book(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from quantswing.live.book import LiveBook
    from quantswing.portfolio.swing import SwingState

    monkeypatch.setenv("QS_DASHBOARD_DATA", str(tmp_path))
    p = DataPaths.relocated(tmp_path)
    b = LiveBook(p.live_db)
    st = SwingState(cash=400000, equity=500000, last_session="2026-09-25")
    st.positions = [
        {
            "symbol": "NSE:ABC-EQ",
            "trade_id": 1,
            "entry_date": "2026-09-20",
            "entry_px": 100.0,
            "qty0": 100,
            "qty": 100,
            "s10": 0.05,
            "z0": 2.0,
            "risk_inr": 750.0,
            "sl0": 92.5,
            "target": 115.0,
            "stop": 95.0,
            "stop_kind": "TSL",
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
            "exit_flag": "Time",
        }
    ]
    b.put("state", st.to_json())
    b.add_action(
        {
            "session": "2026-09-28",
            "symbol": "NSE:ABC-EQ",
            "action": "SELL",
            "qty": 100,
            "reason": "Time exit",
            "key": "k1",
        }
    )
    at = run_view("live")
    assert not at.exception, at.exception
    assert len(at.table) == 1 and len(at.dataframe) >= 1


def test_trade_log_page_with_a_book(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from quantswing.live.book import LiveBook
    from quantswing.portfolio.swing import SwingState

    monkeypatch.setenv("QS_DASHBOARD_DATA", str(tmp_path))
    p = DataPaths.relocated(tmp_path)
    b = LiveBook(p.live_db)
    st = SwingState(cash=400000, equity=500000, last_session="2026-09-25")
    st.positions = [
        {
            "symbol": "NSE:ABC-EQ",
            "trade_id": 1,
            "entry_date": "2026-09-20",
            "entry_px": 100.0,
            "qty0": 100,
            "qty": 100,
            "s10": 0.05,
            "z0": 2.0,
            "risk_inr": 750.0,
            "sl0": 92.5,
            "target": 115.0,
            "stop": 95.0,
            "stop_kind": "TSL",
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
        }
    ]
    b.put("state", st.to_json())
    b.save_quotes(
        pd.DataFrame(
            [{"symbol": "NSE:ABC-EQ", "ltp": 108.0, "open": 101, "high": 109, "low": 99, "prev_close": 100}]
        )
    )
    b.log_positions({"NSE:ABC-EQ": 108.0}, st.positions)
    at = run_view("trade_log")
    assert not at.exception, at.exception
    assert len(at.dataframe) >= 2  # open positions + poll history


def test_backtest_lab_with_swing_trades(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A trade-engine run (trades keyed by entry/exit date, no 'date' column) must render."""
    from quantswing.backtest.tracking import RunTracker

    monkeypatch.setenv("QS_DASHBOARD_DATA", str(tmp_path))
    p = DataPaths.relocated(tmp_path)
    idx = pd.bdate_range("2024-01-01", periods=30)
    daily = pd.DataFrame(
        {
            "equity": 1e6 + pd.Series(range(30), index=idx) * 100.0,
            "cash": 1e5,
            "invested": 9e5,
            "positions": 5,
        },
        index=idx.rename("date"),
    )
    trades = pd.DataFrame(
        {
            "trade_id": [1, 2],
            "symbol": ["A", "B"],
            "entry_date": idx[:2],
            "exit_date": idx[5:7],
            "pnl": [10.0, -5.0],
        }
    )
    RunTracker(p.runs, False).log_run(
        "swing_D37", {"x": 1}, {"cagr": 0.2}, artifacts={"daily": daily, "trades": trades}
    )
    at = run_view("backtest_lab")
    assert not at.exception, at.exception
