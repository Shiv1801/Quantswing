"""Labels, uniqueness, metrics, DSR, Monte Carlo, walk-forward, tracker."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantswing.alpha.labels import triple_barrier, uniqueness_weights
from quantswing.backtest import dsr, metrics, montecarlo
from quantswing.backtest.tracking import RunTracker
from quantswing.backtest.walkforward import purged_walk_forward


def panels(
    paths: dict[str, list[float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(next(iter(paths.values())))
    idx = pd.bdate_range("2024-01-01", periods=n)
    c = pd.DataFrame(paths, index=idx)
    return c, c * 1.001, c * 0.999, pd.DataFrame(0.01, index=idx, columns=c.columns)


def test_triple_barrier_outcomes() -> None:
    # R = 2·0.01·100 = 2 → upper 103, lower 98
    c, h, lo, sig = panels(
        {"UP": [100, 101, 104, 104], "DN": [100, 99, 97, 97], "FLAT": [100, 100.5, 100.2, 100.4]}
    )
    h.loc[h.index[1], "FLAT"] = 103.5  # both barriers on bar 1
    lo.loc[lo.index[1], "FLAT"] = 97.5
    ev = pd.DataFrame({"date": [c.index[0]] * 3, "symbol": ["UP", "DN", "FLAT"]})
    lab = triple_barrier(ev, c, h, lo, sig, 1.5, 2.0, 3).set_index("symbol")
    assert lab.at["UP", "label"] == 1 and lab.at["UP", "holding"] == 2
    assert lab.at["DN", "label"] == -1
    assert lab.at["FLAT", "label"] == -1  # worst case on a two-sided bar
    c2, h2, lo2, s2 = panels({"V": [100, 100.5, 101, 101.2]})
    v = triple_barrier(
        pd.DataFrame({"date": [c2.index[0]], "symbol": ["V"]}), c2, h2, lo2, s2, 1.5, 2.0, 3
    )
    assert v.iloc[0]["label"] == 1 and v.iloc[0]["holding"] == 3  # vertical, positive sign


def test_uniqueness() -> None:
    idx = pd.bdate_range("2024-01-01", periods=6)
    lab = pd.DataFrame({"symbol": ["A", "A"], "date": [idx[0], idx[2]], "t_end": [idx[3], idx[5]]})
    w = uniqueness_weights(lab, idx)
    assert w.iloc[0] == pytest.approx((1 + 1 + 0.5 + 0.5) / 4)
    assert w.iloc[1] == pytest.approx((0.5 + 0.5 + 1 + 1) / 4)


def test_portfolio_and_trade_metrics() -> None:
    idx = pd.bdate_range("2024-01-01", periods=253)
    eq = pd.Series(100 * 1.001 ** np.arange(253), index=idx)
    eq.iloc[100:110] *= 0.9
    daily = pd.DataFrame({"equity": eq, "invested": eq * 0.5, "positions": 3})
    m = metrics.portfolio_metrics(daily, None, 0.0)
    assert m["cagr"] == pytest.approx(1.001**252 - 1, rel=1e-6)
    assert m["max_dd"] == pytest.approx(-0.1, abs=0.002) and m["exposure"] == pytest.approx(0.5)
    tr = pd.DataFrame(
        {
            "r_mult": [2.0, -1.0, 1.0, -1.0],
            "net_pnl": [200, -100, 100, -100],
            "capture": [0.5, np.nan, 0.4, np.nan],
            "holding": [10, 5, 8, 3],
            "costs": [1, 1, 1, 1],
            "exit_date": pd.to_datetime(["2024-01-05", "2024-06-01", "2025-01-01", "2025-02-01"]),
        }
    )
    t = metrics.trade_metrics(tr)
    assert t["win_rate"] == 0.5 and t["expectancy_r"] == 0.25 and t["profit_factor"] == 1.5
    assert metrics.max_year_share(tr) == pytest.approx(100 / 100)


def test_dsr_properties() -> None:
    assert dsr.psr(0.1, 0.1, 0.0, 3.0, 500) == pytest.approx(0.5)
    assert dsr.expected_max_sharpe(100, 0.001) > dsr.expected_max_sharpe(10, 0.001) > 0
    good = dsr.deflated_sharpe(0.1, 0.0, 3.0, 1000, 1, 0.0)
    many = dsr.deflated_sharpe(0.1, 0.0, 3.0, 1000, 500, 0.002)
    assert good > 0.99 and many < good


def test_montecarlo() -> None:
    mc = montecarlo.simulate(np.full(50, 0.01), 200, 1)
    assert mc.dd_p95_reshuffle == 0.0
    mix = montecarlo.simulate(np.r_[np.full(30, 0.02), np.full(20, -0.02)], 2000, 1)
    assert 0 < mix.dd_p50_reshuffle <= mix.dd_p95_reshuffle


def test_purged_walk_forward() -> None:
    sess = pd.bdate_range("2016-01-01", "2021-12-31")
    starts = pd.Series(sess[::5])
    ends = pd.Series([sess[min(sess.get_loc(s) + 20, len(sess) - 1)] for s in starts])
    folds = purged_walk_forward(starts, ends, sess, 3, 1, 20)
    assert [f.test_start.year for f in folds] == [2019, 2020, 2021]
    for f in folds:
        tr_end = ends.iloc[f.train_idx].max()
        assert tr_end < f.test_start  # purged: no training label runs into the test window
        assert starts.iloc[f.test_idx].min() >= f.test_start


def test_tracker(tmp_path: Path) -> None:
    tk = RunTracker(tmp_path)
    tk.log_run("a", {"x": 1}, {"sharpe_pp": 0.05})
    tk.log_run("b", {"x": 2}, {"sharpe_pp": 0.07}, artifacts={"t": pd.DataFrame({"a": [1]})})
    n, v = tk.trial_stats()
    assert n == 2 and v == pytest.approx(np.var([0.05, 0.07], ddof=1))
    assert len(tk.runs()) == 2 and not math.isnan(v)
