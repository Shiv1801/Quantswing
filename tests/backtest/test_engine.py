"""Engine scenarios on scripted price paths: entry, partial, breakeven, trail, stops, gaps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantswing.backtest.engine import Backtester, MarketData
from quantswing.core.config import get_config

CFG = get_config()
EX = CFG.backtest.exits


def md_from(
    close: list[float],
    high: list[float] | None = None,
    low: list[float] | None = None,
    open_: list[float] | None = None,
    sigma: float = 0.02,
) -> tuple[MarketData, pd.DataFrame, pd.DataFrame]:
    n = len(close)
    idx = pd.bdate_range("2024-01-01", periods=n)
    df = lambda v: pd.DataFrame({"A": v, "B": [100.0] * n}, index=idx)  # noqa: E731
    c = df(close)
    o = df(open_ or close)
    h = df(high or [x * 1.001 for x in close])
    lo = df(low or [x * 0.999 for x in close])
    sig = pd.DataFrame(sigma, index=idx, columns=["A", "B"])
    adv = pd.DataFrame(1e12, index=idx, columns=["A", "B"])  # capacity never binds
    md = MarketData(o, h, lo, c, sig, sig, sig * 0, adv)
    scores = pd.DataFrame({"A": [5.0] + [np.nan] * (n - 1), "B": np.nan}, index=idx)
    gate = pd.DataFrame(True, index=idx, columns=["A", "B"])
    return md, scores, gate


def run(
    md: MarketData, scores: pd.DataFrame, gate: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = Backtester(md, scores, gate, CFG).run()
    return r.trades, r.daily


def test_entry_next_open_and_stop_loss() -> None:
    # entry day 1 @100 → SL = 100·(1 − 2·0.02) = 96; day 3 low 95 → stopped at 96
    close = [100, 100, 99, 95.5, 95]
    low = [99.9, 99.9, 98.9, 95.0, 94.9]
    open_ = [100, 100, 99, 99, 95]
    md, s, g = md_from(close, low=low, open_=open_)
    tr, _ = run(md, s, g)
    t = tr.iloc[0]
    assert t["entry_date"] == md.close.index[1] and t["entry_px"] == pytest.approx(
        100.05
    )  # +0.05% spread
    assert t["reason"] == "sl" and t["exit_px"] == pytest.approx(96.048 * (1 - 0.0005), rel=1e-4)
    assert t["r_mult"] == pytest.approx(-1.0, abs=0.08)  # −1R plus costs
    assert len(tr) == 1


def test_gap_through_stop_fills_at_open() -> None:
    close = [100, 100, 100, 90, 90]
    open_ = [100, 100, 100, 90, 90]
    md, s, g = md_from(close, open_=open_, low=[99.9, 99.9, 99.9, 89.9, 89.9])
    tr, _ = run(md, s, g)
    assert tr.iloc[0]["reason"] == "sl_gap"
    assert tr.iloc[0]["exit_px"] == pytest.approx(90 * (1 - 0.0005))
    assert tr.iloc[0]["r_mult"] < -2  # gap loss exceeds 1R


def test_partial_then_breakeven_same_bar_worst_case() -> None:
    # target +1.5R = 100 + 1.5·4 = 106; day 2 high 107 and low 99 → partial then BE stop
    close = [100, 100, 101, 101]
    high = [100.1, 100.1, 107, 101.1]
    low = [99.9, 99.9, 99.0, 100.9]
    md, s, g = md_from(close, high=high, low=low)
    tr, _ = run(md, s, g)
    t = tr.iloc[0]
    assert t["partial"] and t["reason"] == "breakeven"
    assert t["gross_pnl"] > 0  # half booked at +1.5R, rest ~flat


def test_trailing_stop_ratchets_and_exits() -> None:
    up = [100 * 1.02**k for k in range(15)]
    close = [100.0, *up, up[-1] * 0.9, up[-1] * 0.9]
    md, s, g = md_from(close)
    tr, _ = run(md, s, g)
    t = tr.iloc[0]
    assert t["reason"] in ("trail", "trail_gap") and t["partial"]
    assert t["r_mult"] > 1.5 and t["mfe_r"] > t["r_mult"]


def test_time_stop() -> None:
    close = [100.0] * (EX.time_stop_sessions + 5)
    md, s, g = md_from(close)
    tr, _ = run(md, s, g)
    assert tr.iloc[0]["reason"] == "time"
    assert tr.iloc[0]["holding"] == EX.time_stop_sessions


def test_sizing_respects_risk_and_equity_accounting() -> None:
    close = [100.0] * 5
    md, s, g = md_from(close)
    tr, daily = run(md, s, g)
    t = tr.iloc[0]
    eq0 = CFG.risk.paper_capital_inr
    risk = t["qty"] * t["unit"]
    assert risk <= eq0 * CFG.risk.risk_per_trade * 1.001
    assert t["qty"] * t["entry_px"] <= eq0 * CFG.risk.single_name_max + 1
    # day of entry: equity = capital − buy costs − spread paid (marked at close 100)
    assert daily["equity"].iloc[1] == pytest.approx(
        eq0 - t["qty"] * (t["entry_px"] - 100) - (t["costs"] - 0), rel=1e-3
    )
    assert daily["equity"].iloc[0] == eq0


def test_no_signal_no_trade_and_gate_blocks() -> None:
    md, s, g = md_from([100.0] * 5)
    g.loc[:, "A"] = False
    tr, daily = run(md, s, g)
    y = (1 + CFG.costs.idle_cash_annual_yield) ** (1 / 252)
    expect = CFG.risk.paper_capital_inr * y ** np.arange(len(daily))
    assert tr.empty and np.allclose(daily["equity"].to_numpy(), expect)


def test_drawdown_pause_lifts_after_peak_rolls_off() -> None:
    rk = CFG.risk.model_copy(update={"dd_peak_window_sessions": 5, "dd_stop_new_entries": 0.001})
    cfg = CFG.model_copy(update={"risk": rk})
    close = [100.0, 100.0, 90.0, 90.0] + [90.0] * 12
    md, s, g = md_from(close, open_=list(close), low=[c * 0.999 for c in close])
    s.loc[:, "A"] = 5.0  # always a signal
    res = Backtester(md, s, g, cfg).run()
    assert res.daily["halted"].any()
    assert not res.daily["halted"].iloc[-1]  # the old peak rolled off the window
    assert len(res.trades) >= 2  # trading resumed
