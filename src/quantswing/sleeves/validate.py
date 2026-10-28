"""Per-sleeve purged walk-forward validation (05 §3 "Sleeve test").

For each test year Y (train = the ``train_years`` before Y):
1. Training events = sleeve signals dated in the train window whose label horizon ends
   ``embargo`` sessions before Y starts (purging + embargo).
2. Fit exits from their paths (:mod:`exit_fit`), label them with the fitted barriers,
   fit p(win)/EV on the composite score (:mod:`alpha.calibration`).
3. Run the engine over Y only, with fitted exits, entries ranked by EV (EV > 0), the
   sleeve's stop-entry levels / confirmation, at 1× and ``cost_stress_multiplier``× costs.
Test-year returns are chained into one out-of-sample curve. Verdict ON iff OOS Sharpe ≥
``sharpe_min_oos``, OOS trades ≥ ``min_oos_trades`` and 2×-cost expectancy >
``stress_expectancy_min``; otherwise OFF (05 §3, P5 of 01).
A final fit on the most recent window gives the parameters used live (P6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantswing.alpha.calibration import EvModel, fit_ev
from quantswing.alpha.labels import triple_barrier
from quantswing.backtest import metrics
from quantswing.backtest.engine import Backtester, BacktestResult, MarketData
from quantswing.core.config import AppConfig, ExitsCfg
from quantswing.execution.costs import CostModel
from quantswing.factors.common import Panels
from quantswing.sleeves import registry
from quantswing.sleeves.base import FactorView, SleeveSignals
from quantswing.sleeves.exit_fit import FitResult, event_paths, fit_exits

NOTIONAL = 100_000.0
REF_PRICE = 1_000.0


def round_trip_cost_frac(cfg: AppConfig) -> float:
    """Round-trip charges + 2× half-spread as a fraction of notional."""
    cm = CostModel(cfg.costs)
    q = NOTIONAL / REF_PRICE
    fees = cm.fill("buy", q, REF_PRICE).total + cm.fill("sell", q, REF_PRICE).total
    return fees / NOTIONAL + 2 * cfg.costs.slippage.default_half_spread_pct


@dataclass
class Fitted:
    """Parameters fitted on one training window."""

    fit: FitResult
    ev: EvModel
    n_train: int


def fit_window(
    events: pd.DataFrame, p: Panels, md: MarketData, scores: pd.DataFrame, cfg: AppConfig
) -> Fitted:
    """Fit exits + EV on ``events`` (already purged)."""
    fc = cfg.sleeves.fit
    paths = event_paths(
        events, p.open, p.high, p.low, p.close, md.sigma_sl, fc.horizon, fc.winner_horizon
    )
    fit = fit_exits(paths, fc, cfg.backtest.exits)
    ex = fit.exits
    lab = triple_barrier(
        events,
        p.close,
        p.high,
        p.low,
        md.sigma_sl,
        ex.partial_at_r,
        ex.k_sl_sigma,
        ex.max_hold_sessions,
    )
    if lab.empty:
        return Fitted(fit, fit_ev(np.array([]), np.array([]), np.array([])), 0)
    ret_r = (np.exp(lab["ret"].to_numpy()) - 1) * lab["p0"].to_numpy() / lab["unit"].to_numpy()
    sc = scores.stack(future_stack=True)
    x = sc.reindex(pd.MultiIndex.from_arrays([lab["date"], lab["symbol"]])).to_numpy(
        dtype="float64"
    )
    return Fitted(fit, fit_ev(x, lab["label"].to_numpy(), ret_r), len(lab))


def ev_panel(
    sig: SleeveSignals, scores: pd.DataFrame, md: MarketData, fitted: Fitted, cost_frac: float
) -> pd.DataFrame:
    """EV (R units) where the sleeve signals, NaN elsewhere."""
    k = fitted.fit.exits.k_sl_sigma
    cost_r = (cost_frac / (k * md.sigma_sl)).to_numpy(dtype="float64")
    ev = fitted.ev.ev(scores.reindex_like(sig.signal).to_numpy(dtype="float64"), cost_r)
    out = pd.DataFrame(ev, index=sig.signal.index, columns=sig.signal.columns)
    return out.where(sig.signal.fillna(False).astype(bool))


def _purged(
    events: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    horizon: int,
    embargo: int,
) -> pd.DataFrame:
    pos = pd.Series(np.arange(len(sessions)), index=sessions)
    cut = int(sessions.searchsorted(t1)) - embargo - horizon
    ev = events[(events["date"] >= t0) & (events["date"] < t1)]
    return ev[pos.loc[ev["date"]].to_numpy() < cut]


@dataclass
class SleeveResult:
    """Walk-forward outcome of one sleeve."""

    name: str
    n_signals: int
    folds: pd.DataFrame
    daily: pd.DataFrame
    trades: pd.DataFrame
    stress_trades: pd.DataFrame
    port: dict[str, float]
    trade: dict[str, float]
    stress: dict[str, float]
    verdict: str
    reasons: list[str]
    live_exits: dict[str, Any]
    live_ev: dict[str, float]
    live_confident: bool
    extra: dict[str, Any] = field(default_factory=dict)


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    parts = [f for f in frames if not f.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _chain(results: list[BacktestResult], capital: float) -> pd.DataFrame:
    parts = []
    for r in results:
        if r.daily.empty:
            continue
        d = r.daily.copy()
        d["ret"] = d["equity"].pct_change().fillna(d["equity"].iloc[0] / capital - 1)
        d["expo"] = d["invested"] / d["equity"]
        parts.append(d[["ret", "expo", "positions", "halted"]])
    if not parts:
        return pd.DataFrame(columns=["equity", "invested", "positions", "halted"])
    df = pd.concat(parts)
    df["equity"] = capital * (1 + df["ret"]).cumprod()
    df["invested"] = df["expo"] * df["equity"]
    return df[["equity", "invested", "positions", "halted"]]


def run_sleeve(
    name: str,
    p: Panels,
    md: MarketData,
    f: FactorView,
    gate: pd.DataFrame,
    scores: pd.DataFrame,
    cfg: AppConfig,
    regime: pd.DataFrame | None = None,
) -> SleeveResult:
    """Walk-forward test one sleeve and fit its live parameters."""
    bt, vc, sv = cfg.backtest, cfg.backtest.validation, cfg.sleeves.validation
    sig = registry.build(name, p, f, gate, cfg)
    events = sig.events()
    sessions = pd.DatetimeIndex(p.close.index)
    years = sorted(set(sessions.year))
    cost_frac = round_trip_cost_frac(cfg)
    entry = bt.baseline.model_copy(update={"min_score": 0.0, "name": name})
    horizon = cfg.sleeves.fit.horizon
    expo = None
    if regime is not None and "mult" in regime:
        expo = regime["mult"].astype(float)
        if name != "S2":  # 03 §5 tilt: Bear → minimal S2 only
            expo = expo.where(~regime["state"].isin(["bear", "crisis"]), 0.0)
    fold_rows: list[dict[str, Any]] = []
    base_runs: list[BacktestResult] = []
    stress_runs: list[BacktestResult] = []
    for k in range(vc.train_years, len(years)):
        y = years[k]
        t0 = pd.Timestamp(f"{years[k - vc.train_years]}-01-01")
        test = sessions[sessions.year == y]
        if len(test) == 0:
            continue
        tr = _purged(events, sessions, t0, test[0], horizon, vc.embargo_sessions)
        fitted = fit_window(tr, p, md, scores, cfg)
        evp = ev_panel(sig, scores, md, fitted, cost_frac)
        kw: dict[str, Any] = dict(
            exits=fitted.fit.exits,
            entry=entry,
            entry_level=sig.entry_level,
            confirm=sig.confirm,
            label=name,
            exposure=expo,
            delisted_last=p.delisted_last,
            delisted_haircut=p.delisted_haircut,
        )
        res = Backtester(md, evp, gate, cfg, **kw).run(start=test[0], end=test[-1])
        st = Backtester(md, evp, gate, cfg, bt.execution.cost_stress_multiplier, **kw).run(
            start=test[0], end=test[-1]
        )
        base_runs.append(res)
        stress_runs.append(st)
        tm = metrics.trade_metrics(res.trades)
        fold_rows.append(
            {
                "test_year": y,
                "train_events": len(tr),
                "fit_ok": fitted.fit.confident,
                "k_sl": fitted.fit.exits.k_sl_sigma,
                "partial_r": fitted.fit.exits.partial_at_r,
                "trail_k": fitted.fit.exits.trail_k_below_switch,
                "time_stop": fitted.fit.exits.time_stop_sessions,
                "p_slope": fitted.ev.logit.b,
                "trades": tm.get("trades", 0.0),
                "expectancy_r": tm.get("expectancy_r", float("nan")),
                "ret": (
                    float(res.daily["equity"].iloc[-1] / res.daily["equity"].iloc[0] - 1)
                    if not res.daily.empty
                    else float("nan")
                ),
            }
        )
    capital = float(cfg.risk.paper_capital_inr)
    daily = _chain(base_runs, capital)
    trades = _concat([r.trades for r in base_runs])
    s_trades = _concat([r.trades for r in stress_runs])
    port = metrics.portfolio_metrics(daily, p.bench, bt.risk_free_annual) if len(daily) > 2 else {}
    trade = metrics.trade_metrics(trades)
    stress = metrics.trade_metrics(s_trades)
    reasons: list[str] = []
    sharpe = port.get("sharpe", float("nan"))
    if not (np.isfinite(sharpe) and sharpe >= sv.sharpe_min_oos):
        reasons.append(f"OOS Sharpe {sharpe:.2f} < {sv.sharpe_min_oos}")
    if trade.get("trades", 0) < sv.min_oos_trades:
        reasons.append(f"only {trade.get('trades', 0):.0f} OOS trades (< {sv.min_oos_trades})")
    se = stress.get("expectancy_r", float("nan"))
    if not (np.isfinite(se) and se > sv.stress_expectancy_min):
        reasons.append(f"2x-cost expectancy {se:+.2f}R <= {sv.stress_expectancy_min}")
    verdict = "ON" if not reasons else "OFF"
    # live parameters: most recent train window, purged at the last session
    t_end = sessions[-1] + pd.Timedelta(days=1)
    live = fit_window(
        _purged(
            events, sessions, pd.Timestamp(f"{years[-1] - vc.train_years}-01-01"), t_end, horizon, 0
        ),
        p,
        md,
        scores,
        cfg,
    )
    live_ex: ExitsCfg = live.fit.exits
    return SleeveResult(
        name=name,
        n_signals=len(events),
        folds=pd.DataFrame(fold_rows),
        daily=daily,
        trades=trades,
        stress_trades=s_trades,
        port=port,
        trade=trade,
        stress=stress,
        verdict=verdict,
        reasons=reasons,
        live_exits=live_ex.model_dump(),
        live_confident=live.fit.confident,
        live_ev={
            "a": live.ev.logit.a,
            "b": live.ev.logit.b,
            "mu": live.ev.logit.mu,
            "sd": live.ev.logit.sd,
            "avg_win_r": live.ev.avg_win_r,
            "avg_loss_r": live.ev.avg_loss_r,
            "n": float(live.ev.n),
        },
    )
