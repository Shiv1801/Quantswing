"""P4: sleeve rules, exit fitting, calibration, stop-entry engine, walk-forward end-to-end."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantswing.alpha.calibration import fit_ev, fit_logistic
from quantswing.backtest.engine import Backtester
from quantswing.core.config import get_config
from quantswing.sleeves import s1_breakout, s2_pullback, s3_ath_recovery, s4_high_momentum
from quantswing.sleeves.base import FactorView, rising_edge, xs_pct_rank
from quantswing.sleeves.exit_fit import EventPaths, fit_exits
from quantswing.store.repositories import DataPaths
from tests.backtest.test_engine import md_from
from tests.factors.test_factors import synth_panels

CFG = get_config()
R = CFG.sleeves.rules


def const(p, v: float) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    return pd.DataFrame(v, index=p.close.index, columns=p.close.columns)


def test_xs_rank_and_edge() -> None:
    idx = pd.bdate_range("2024-01-01", periods=3)
    x = pd.DataFrame({"A": [1, 2, 3], "B": [3, 2, 1], "C": [2, 2, 2]}, index=idx, dtype=float)
    r = xs_pct_rank(x)
    assert r.loc[idx[0], "B"] == 1.0 and r.loc[idx[0], "A"] == pytest.approx(1 / 3)
    e = rising_edge(pd.DataFrame({"A": [False, True, True]}, index=idx))
    assert list(e["A"]) == [False, True, False]


def test_s1_rules() -> None:
    p = synth_panels(t=300, n=6)
    f = FactorView(
        p,
        preset={
            "CMP": const(p, 0.1),
            "RNG": const(p, 1.0),
            "TQ120": const(p, 0.2),
            "HURST": const(p, 0.6),
            "VOLZ": const(p, 2.5),
        },
    )
    f._cache["RNG"].iloc[:, 0] = 0.1  # S0 is the tightest range → bottom tercile
    gate = const(p, 1).astype(bool)
    s = s1_breakout.build(p, f, gate, R.S1, 60)
    assert s.signal.iloc[-1, 0] and not s.signal.iloc[-1, 5]
    lvl = s.entry_level.iloc[-1, 0]  # type: ignore[union-attr]
    assert lvl > p.high.iloc[-20:, 0].max()
    assert s.confirm is not None and bool(s.confirm.iloc[-1, 0])


def test_s2_rules_require_cross_after_setup() -> None:
    p = synth_panels(t=300, n=4)
    kal = const(p, 0.5)
    kal.iloc[-1] = 1.5  # crosses today
    dev = const(p, -2.0)
    f = FactorView(
        p,
        preset={
            "TQ120": const(p, 1.0),
            "R2_120": const(p, 0.9),
            "DEV": dev,
            "OUHL": const(p, 5.0),
            "VOLZ": const(p, 0.0),
            "KAL_T": kal,
        },
    )
    f._cache["TQ120"].iloc[:, 1:] = 0.0  # only S0 is top tercile
    s = s2_pullback.build(p, f, const(p, 1).astype(bool), R.S2)
    assert s.signal.iloc[-1, 0] and not s.signal.iloc[-1, 1] and not s.signal.iloc[-2, 0]


def test_s3_and_s4_rules() -> None:
    p = synth_panels(t=400, n=5)
    dd = const(p, -0.05)
    dd.iloc[200:260, 0] = -0.3
    slope = pd.DataFrame(
        np.tile(np.linspace(0, 0.01, 400)[:, None], (1, 5)),
        index=p.close.index,
        columns=p.close.columns,
    )
    f = FactorView(
        p,
        preset={
            "DD": dd,
            "RR": const(p, 0.8),
            "HURST": const(p, 0.6),
            "KAL_SLOPE": slope,
            "KAL_T": const(p, 2.0),
        },
    )
    s3 = s3_ath_recovery.build(p, f, const(p, 1).astype(bool), R.S3)
    assert s3.signal.iloc[:, 0].sum() >= 1 and s3.signal.iloc[:, 1].sum() == 0
    p4 = synth_panels(t=400, n=5)
    close = p4.close.copy()
    close.iloc[-1, 0] = close.iloc[:-1, 0].max() * 1.05  # new closing high
    p4 = type(p4)(p4.open, p4.high, p4.low, close, p4.volume, p4.turnover, p4.bench)
    mom = const(p4, 0.0)
    mom.iloc[:, 0] = 5.0
    f4 = FactorView(
        p4,
        preset={
            "H52": const(p4, 0.99),
            "MOM126": mom,
            "RMOM": const(p4, 1.0),
            "VOLZ": const(p4, 2.0),
        },
    )
    s4 = s4_high_momentum.build(p4, f4, const(p4, 1).astype(bool), R.S4)
    assert s4.signal.iloc[-1, 0] and not s4.signal.iloc[-1, 1]


def test_fit_exits_from_paths() -> None:
    rng = np.random.default_rng(0)
    n, h = 2000, CFG.sleeves.fit.horizon
    drift = np.linspace(0, 3, h)
    close_k = drift + rng.normal(0, 0.5, (n, h))
    winners = close_k[:, CFG.sleeves.fit.winner_horizon - 1] > 0
    mae = np.abs(rng.normal(1.0, 0.3, n))
    mfe = close_k.max(axis=1) + 0.5
    res = fit_exits(EventPaths(mae, mfe, close_k, winners), CFG.sleeves.fit, CFG.backtest.exits)
    ex = res.exits
    assert res.confident
    lo, hi = CFG.sleeves.fit.k_sl_bounds
    assert lo <= ex.k_sl_sigma <= hi and ex.k_sl_sigma == pytest.approx(
        np.percentile(mae[winners], 90) * 1.1, rel=0.05
    )
    assert (
        CFG.sleeves.fit.partial_r_bounds[0]
        <= ex.partial_at_r
        <= CFG.sleeves.fit.partial_r_bounds[1]
    )
    assert ex.trail_k_above_switch <= ex.trail_k_below_switch
    assert ex.max_hold_sessions >= ex.time_stop_sessions
    few = fit_exits(
        EventPaths(mae[:5], mfe[:5], close_k[:5], winners[:5]), CFG.sleeves.fit, CFG.backtest.exits
    )
    assert not few.confident and few.exits == CFG.backtest.exits


def test_logistic_and_ev() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=5000)
    y = (rng.random(5000) < 1 / (1 + np.exp(-(0.2 + 0.8 * x)))).astype(float)
    lg = fit_logistic(x, y)
    assert lg.b * lg.sd == pytest.approx(0.8, abs=0.1) or lg.b == pytest.approx(
        0.8 * lg.sd, abs=0.1
    )
    assert fit_logistic(np.ones(50), y[:50]).b == 0.0
    label = np.where(y == 1, 1, -1)
    ret_r = np.where(y == 1, 1.5, -1.0)
    ev = fit_ev(x, label, ret_r)
    hi_ev, lo_ev = (
        ev.ev(np.array([2.0]), np.array([0.1]))[0],
        ev.ev(np.array([-2.0]), np.array([0.1]))[0],
    )
    assert hi_ev > 0 > lo_ev and ev.avg_win_r == 1.5 and ev.avg_loss_r == 1.0


def test_engine_stop_entry_and_confirm() -> None:
    close = [100.0, 100.0, 101.0, 103.0, 103.0]
    high = [100.5, 100.5, 101.5, 104.0, 103.5]
    open_ = [100.0, 100.0, 101.0, 101.5, 103.0]  # day 3 opens below the level, trades through it
    md, s, g = md_from(close, high=high, low=[c * 0.995 for c in close], open_=open_)
    s.loc[:, "A"] = 1.0
    lvl = pd.DataFrame(np.nan, index=s.index, columns=s.columns)
    lvl["A"] = 102.0  # needs high ≥ 102 → first on day 3
    res = Backtester(md, s, g, CFG, entry_level=lvl, label="S1").run()
    t = res.trades.iloc[0]
    assert (
        t["entry_date"] == s.index[3]
        and t["entry_px"] == pytest.approx(102.0 * 1.0005)
        and t["sleeve"] == "S1"
    )
    conf = pd.DataFrame(False, index=s.index, columns=s.columns)
    res2 = Backtester(md, s, g, CFG, entry_level=lvl, confirm=conf).run()
    assert res2.trades.empty


def build_long_root(root: Path) -> Path:
    """Synthetic 5.5-year, 60-stock market through P1 → P2, built under ``root``."""
    from quantswing.alpha.research import build_factors, ic_analysis
    from quantswing.ingest import pipeline
    from quantswing.store.database import build_database
    from tests.conftest import FakeFD, make_candles, symbol_master_csv, weekdays
    from tests.store.test_pipeline import FakeHTTP

    paths = DataPaths.relocated(root)
    dates = weekdays(date(2020, 1, 1), date(2025, 6, 30))
    syms = [f"NSE:S{i:02d}-EQ" for i in range(60)]
    rows = [
        {"name": f"CO{i}", "ticker": s, "isin": f"INE{i:03d}A01011"} for i, s in enumerate(syms)
    ]
    pipeline.refresh_universe(paths, CFG, http=FakeHTTP(symbol_master_csv(rows)))
    series = {s: make_candles(dates, seed=i) for i, s in enumerate(CFG.settings.data.indices)}
    series |= {s: make_candles(dates, seed=500 + i) for i, s in enumerate(syms)}
    fd = FakeFD(series)
    pipeline.load_indices(fd, paths, date(2015, 1, 1), dates[-1], CFG)  # type: ignore[arg-type]
    pipeline.build_calendar(paths, CFG)
    pipeline.load_equities(fd, paths, syms, date(2015, 1, 1), dates[-1], None, CFG)  # type: ignore[arg-type]
    pipeline.validate_all(paths, syms, dates[-1], CFG)
    build_database(paths)
    build_factors(paths, CFG)
    ic_analysis(paths, CFG)
    return root


@pytest.fixture(scope="module")
def long_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_long_root(tmp_path_factory.mktemp("p4"))


def test_walk_forward_all_sleeves(long_root: Path) -> None:
    from quantswing.sleeves.report import write_report
    from quantswing.sleeves.runner import fitted_path, run_all

    paths = DataPaths.relocated(long_root)
    bt = CFG.backtest
    cfg = CFG.model_copy(
        update={
            "backtest": bt.model_copy(
                update={"validation": bt.validation.model_copy(update={"train_years": 2})}
            )
        }
    )
    res = run_all(paths, cfg)
    assert [r.name for r in res] == ["S1", "S2", "S3", "S4"]
    assert all(r.verdict in ("ON", "OFF") for r in res)
    assert sum(r.n_signals for r in res) > 0
    assert any(len(r.folds) >= 2 for r in res)
    assert fitted_path(paths).exists()
    out = write_report(res, paths.reports, ["n"])
    assert "walk-forward" in out.read_text(encoding="utf-8")
    for r in res:  # OOS trades only inside test years
        if not r.trades.empty:
            first_test = pd.Timestamp(f"{2020 + 2}-01-01")
            assert pd.to_datetime(r.trades["entry_date"]).min() >= first_test


def test_planted_edge_is_detected_and_random_is_rejected() -> None:
    """End-to-end power check: S3 signals followed by drift → ON; same signals, no drift → OFF."""
    from quantswing.backtest.baseline import market_data
    from quantswing.factors.common import Panels
    from quantswing.sleeves.validate import run_sleeve

    def build(drift_per_day: float) -> tuple[Panels, pd.DataFrame]:
        rng = np.random.default_rng(7)
        t, n = 1500, 80
        idx = pd.bdate_range("2019-01-01", periods=t)
        cols = [f"S{i}" for i in range(n)]
        r = rng.normal(0, 0.015, (t, n))
        ev = np.zeros((t, n), dtype=bool)
        for j in range(n):
            for d in rng.choice(np.arange(300, t - 30), 5, replace=False):
                ev[d, j] = True
                r[d + 1 : d + 21, j] += drift_per_day
        close = 100 * np.exp(np.cumsum(r, axis=0))
        opn = np.vstack([close[:1], close[:-1]]) * np.exp(rng.normal(0, 0.002, (t, n)))
        high = np.maximum(opn, close) * (1 + rng.uniform(0, 0.005, (t, n)))
        low = np.minimum(opn, close) * (1 - rng.uniform(0, 0.005, (t, n)))
        vol = rng.integers(50_000, 500_000, (t, n)).astype(float)
        df = lambda a: pd.DataFrame(a, index=idx, columns=cols)  # noqa: E731
        p = Panels(
            df(opn),
            df(high),
            df(low),
            df(close),
            df(vol),
            df(close * vol * 100),
            pd.Series(1000.0, index=idx) * np.exp(np.cumsum(rng.normal(0, 0.01, t))),
        )
        return p, df(ev)

    cfg = CFG.model_copy(
        update={
            "backtest": CFG.backtest.model_copy(
                update={"validation": CFG.backtest.validation.model_copy(update={"train_years": 2})}
            )
        }
    )
    out = {}
    for drift in (0.004, 0.0):
        p, ev = build(drift)
        on = ev.astype(float)
        f = FactorView(
            p,
            preset={
                "DD": on * -0.3 + (1 - on) * -0.05,
                "RR": const(p, 0.8),
                "HURST": const(p, 0.6),
                "KAL_SLOPE": on * 0.01,
                "KAL_T": const(p, 2.0),
            },
        )
        # KAL_SLOPE "accelerating" only on event days; DD window min gets −0.3 → make it pass only on events
        f._cache["DD"] = (
            on.rolling(250, min_periods=1).max() * -0.3
            + (1 - on.rolling(250, min_periods=1).max()) * -0.05
        )
        gate = const(p, 1).astype(bool)
        scores = const(p, 0.0)
        out[drift] = run_sleeve("S3", p, market_data(p, cfg), f, gate, scores, cfg)
    edge, rand = out[0.004], out[0.0]
    assert edge.trade["expectancy_r"] > 0.2 and edge.port["sharpe"] > rand.port.get("sharpe", -99)
    assert edge.verdict == "ON", edge.reasons
    assert rand.verdict == "OFF"


def test_missing_factor_gives_clear_error(tmp_path: Path) -> None:
    from quantswing.sleeves.base import MissingFactorError
    from quantswing.store.panels import FactorStore

    p = synth_panels(t=50, n=3)
    fv = FactorView(p, FactorStore(tmp_path))
    with pytest.raises(MissingFactorError, match="run_factors"):
        _ = fv["R2_120"]
