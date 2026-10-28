"""P3 baseline: composite-score longs through the full engine + validation suite.

Steps: build :class:`MarketData` from adjusted panels → load the P2 composite and gate →
run at 1× and ``cost_stress_multiplier``× costs → metrics, year split, walk-forward test
years, Monte Carlo, Deflated Sharpe (trials from the tracker) → log the run → check
reproducibility (a second identical run must give identical metrics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantswing.alpha.labels import triple_barrier
from quantswing.alpha.research import AUX, COMPOSITE
from quantswing.backtest import dsr, metrics, montecarlo
from quantswing.backtest.engine import Backtester, BacktestResult, MarketData
from quantswing.backtest.tracking import RunTracker, fingerprint
from quantswing.core.config import AppConfig, get_config
from quantswing.factors.common import Panels, rolling_sigma
from quantswing.store.panels import FactorStore, load_panels
from quantswing.store.repositories import DataPaths


def market_data(p: Panels, cfg: AppConfig) -> MarketData:
    """Derive σ and ADV panels the engine needs."""
    bt = cfg.backtest
    r = p.r
    return MarketData(
        open=p.open,
        high=p.high,
        low=p.low,
        close=p.close,
        sigma_sl=rolling_sigma(r, bt.exits.sigma_window),
        sigma_trail=rolling_sigma(r, bt.exits.trail_sigma_window),
        sigma_impact=rolling_sigma(r, bt.execution.impact_vol_window),
        adv_value=p.turnover.rolling(
            bt.sizing.adv_window, min_periods=bt.sizing.adv_window
        ).median(),
    )


@dataclass
class BaselineReport:
    """Everything the P3 report and exit test need."""

    base: BacktestResult
    stress: BacktestResult
    port: dict[str, float]
    trade: dict[str, float]
    stress_trade: dict[str, float]
    by_year: pd.Series
    year_share: float
    mc: montecarlo.McResult
    dsr: float
    n_trials: int
    labels: dict[str, float]
    run_id: str
    reproducible: bool
    checks: dict[str, bool] = field(default_factory=dict)  # P3 exit criteria
    quality: dict[str, bool] = field(default_factory=dict)  # 05 §3 strategy tests (info in P3)


def _params(cfg: AppConfig) -> dict[str, Any]:
    return {
        "baseline": cfg.backtest.baseline.model_dump(),
        "exits": cfg.backtest.exits.model_dump(),
        "sizing": cfg.backtest.sizing.model_dump(),
        "risk": cfg.risk.model_dump(mode="json"),
        "costs_version": cfg.costs.version,
        "factors_version": cfg.factors.version,
        "start": str(cfg.backtest.start),
    }


def label_summary(
    p: Panels, scores: pd.DataFrame, gate: pd.DataFrame, cfg: AppConfig, start: pd.Timestamp
) -> dict[str, float]:
    """Triple-barrier outcome mix of every baseline signal (score ≥ min, gated)."""
    lc, bl = cfg.backtest.labels, cfg.backtest.baseline
    sig = (scores.where(gate.astype(bool)).loc[start:] >= bl.min_score).to_numpy()
    rows, cols = np.nonzero(sig)
    sub_idx = scores.loc[start:].index
    ev = pd.DataFrame({"date": sub_idx[rows], "symbol": scores.columns[cols]})
    if ev.empty:
        return {"events": 0.0}
    lab = triple_barrier(
        ev[["date", "symbol"]],
        p.close,
        p.high,
        p.low,
        rolling_sigma(p.r, cfg.backtest.exits.sigma_window),
        lc.upper_r,
        lc.k_sl_sigma,
        lc.max_hold_sessions,
    )
    return {
        "events": float(len(lab)),
        "p_upper": float((lab["label"] == 1).mean()),
        "p_lower": float((lab["label"] == -1).mean()),
        "p_vertical": float((lab["holding"] >= lc.max_hold_sessions).mean()),
        "mean_log_ret": float(lab["ret"].mean()),
        "avg_holding": float(lab["holding"].mean()),
    }


def trade_returns_on_equity(res: BacktestResult) -> np.ndarray:
    """Net P&L of each trade / equity on its entry date (for Monte Carlo)."""
    if res.trades.empty:
        return np.array([])
    eq = res.daily["equity"]
    e0 = eq.reindex(pd.DatetimeIndex(res.trades["entry_date"])).to_numpy()
    return np.asarray(res.trades["net_pnl"].to_numpy() / e0, dtype="float64")


def run_baseline(
    paths: DataPaths, cfg: AppConfig | None = None, limit: int | None = None
) -> BaselineReport:
    """Run the full P3 baseline + validation suite and log it."""
    cfg = cfg or get_config()
    bt, vc = cfg.backtest, cfg.backtest.validation
    store = FactorStore(paths.factors)
    p = load_panels(paths, cfg, limit=limit)
    scores = store.read("composite", COMPOSITE).reindex_like(p.close)
    gate = store.read(AUX, "gate").reindex_like(p.close).fillna(0)
    md = market_data(p, cfg)
    start = pd.Timestamp(bt.start)

    dl = p.delisted_last
    base = Backtester(
        md, scores, gate, cfg, delisted_last=dl, delisted_haircut=p.delisted_haircut
    ).run(start=start)
    again = Backtester(
        md, scores, gate, cfg, delisted_last=dl, delisted_haircut=p.delisted_haircut
    ).run(start=start)
    stress = Backtester(
        md,
        scores,
        gate,
        cfg,
        bt.execution.cost_stress_multiplier,
        delisted_last=dl,
        delisted_haircut=p.delisted_haircut,
    ).run(start=start)

    port = metrics.portfolio_metrics(base.daily, p.bench, bt.risk_free_annual)
    trade = metrics.trade_metrics(base.trades)
    reproducible = fingerprint(port | trade) == fingerprint(
        metrics.portfolio_metrics(again.daily, p.bench, bt.risk_free_annual)
        | metrics.trade_metrics(again.trades)
    )
    stress_trade = metrics.trade_metrics(stress.trades)
    sr, sk, ku, t = metrics.returns_moments(base.daily)
    tracker = RunTracker(paths.runs, bt.tracking.use_mlflow)
    mc = montecarlo.simulate(trade_returns_on_equity(base), vc.montecarlo_sims, vc.montecarlo_seed)
    labels = label_summary(p, scores, gate, cfg, start)
    run_metrics = {
        **port,
        **trade,
        "sharpe_pp": sr,
        "stress_expectancy_r": stress_trade.get("expectancy_r"),
    }
    run_id = tracker.log_run(
        bt.baseline.name,
        _params(cfg),
        metrics.nan_to_none(run_metrics),
        tags={"data_rows": int(p.close.notna().sum().sum()), "symbols": p.close.shape[1]},
        artifacts={"trades": base.trades, "daily": base.daily},
    )
    n_trials, var_sr = tracker.trial_stats()
    d = dsr.deflated_sharpe(sr, sk, ku, t, max(n_trials, 1), var_sr)
    share = metrics.max_year_share(base.trades)
    rep = BaselineReport(
        base=base,
        stress=stress,
        port=port,
        trade=trade,
        stress_trade=stress_trade,
        by_year=metrics.pnl_by_year(base.trades),
        year_share=share,
        mc=mc,
        dsr=d,
        n_trials=n_trials,
        labels=labels,
        run_id=run_id,
        reproducible=reproducible,
    )
    mdd = abs(port.get("max_dd", float("nan")))
    rep.checks = {
        "reproducible (2 identical runs)": reproducible,
        "run logged": bool(run_id),
        "trades generated": trade.get("trades", 0) > 0,
    }
    rep.quality = {
        f"DSR > {vc.dsr_pass}": d > vc.dsr_pass,
        f"MC 95% DD <= {vc.montecarlo_dd_mult_pass}x backtest DD": mc.dd_p95_reshuffle
        <= vc.montecarlo_dd_mult_pass * mdd,
        "2x costs: expectancy > 0": stress_trade.get("expectancy_r", -1) > 0,
        f"no year > {vc.max_year_pnl_share:.0%} of P&L": share <= vc.max_year_pnl_share,
    }
    return rep
