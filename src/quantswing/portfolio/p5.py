"""P5 orchestration: regime series + factor-portfolio variants + validation summary.

Variants (each logged as a tracker trial, so the Deflated Sharpe counts them):
* ``portfolio_no_regime`` — ranked book on the composite, vol-targeted
* ``portfolio_regime``    — same × HMM regime exposure multiplier
Benchmark: Nifty 500 (price) buy-and-hold. OOS window = sessions from ``oos_start``
(after 3y of IC history + HMM warm-up); nothing else is fitted, so earlier years are
reported but not used for the verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quantswing.alpha.events import event_block
from quantswing.alpha.research import AUX, COMPOSITE
from quantswing.backtest import dsr, metrics
from quantswing.backtest.tracking import RunTracker
from quantswing.core.config import AppConfig, get_config
from quantswing.factors.common import Panels
from quantswing.portfolio.construction import PortfolioResult, run_portfolio
from quantswing.regime.breadth import breadth
from quantswing.regime.hmm import features, run_regime
from quantswing.store.panels import FactorStore, load_panels
from quantswing.store.repositories import DataPaths, connect, write_parquet_frame

REGIME_FILE = "regime.parquet"


def build_regime(p: Panels, gate: pd.DataFrame, paths: DataPaths, cfg: AppConfig) -> pd.DataFrame:
    """Compute and persist the regime series."""
    rc = cfg.risk.regime
    br = breadth(p.logp, gate, rc.breadth_window, rc.breadth_min_names)
    vix = None
    if paths.duckdb.exists():
        con = connect(paths.duckdb, read_only=True)
        try:
            v = con.execute(
                "SELECT date, close FROM indices_daily WHERE symbol = ? ORDER BY date",
                [rc.vix_symbol],
            ).fetchdf()
        finally:
            con.close()
        if not v.empty:
            vix = v.set_index(pd.to_datetime(v["date"]))["close"].reindex(p.close.index)
    feat = features(p.bench, vix, br, rc)
    reg = run_regime(feat, rc, cfg.risk.regime_exposure_mult)
    reg["breadth"] = br
    out = reg.reset_index(names="date")
    write_parquet_frame(out, paths.factors.parent / REGIME_FILE)
    return reg


def load_regime(paths: DataPaths) -> pd.DataFrame | None:
    """Stored regime series (or None)."""
    f = paths.factors.parent / REGIME_FILE
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    return df.set_index(pd.DatetimeIndex(df.pop("date")))


@dataclass
class Variant:
    """One portfolio variant's results."""

    name: str
    res: PortfolioResult
    stress: PortfolioResult
    full: dict[str, float]
    oos: dict[str, float]
    oos_stress: dict[str, float]
    by_year: pd.Series
    turnover: float
    dsr: float
    run_id: str = ""
    checks: dict[str, bool] = field(default_factory=dict)


def _slice(daily: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    return daily[daily.index >= start]


def _turnover(res: PortfolioResult) -> float:
    if res.trades.empty or res.daily.empty:
        return float("nan")
    yrs = len(res.daily) / 252
    return float(res.trades["value"].sum() / res.daily["equity"].mean() / yrs)


def run_p5(
    paths: DataPaths, cfg: AppConfig | None = None, limit: int | None = None
) -> tuple[list[Variant], dict[str, float], pd.DataFrame]:
    """Regime + variants. Returns (variants, benchmark OOS metrics, regime frame)."""
    cfg = cfg or get_config()
    bt = cfg.backtest
    store = FactorStore(paths.factors)
    p = load_panels(paths, cfg, limit=limit)
    gate = store.read(AUX, "gate").reindex_like(p.close).fillna(0).astype(bool)
    scores = store.read("composite", COMPOSITE).reindex_like(p.close)
    reg = build_regime(p, gate, paths, cfg)
    start = pd.Timestamp(bt.start)
    first_comp = scores.dropna(how="all").index.min()
    oos_start = (
        max(start, pd.Timestamp(f"{first_comp.year + bt.validation.train_years}-01-01"))
        if pd.notna(first_comp)
        else start
    )
    tracker = RunTracker(paths.runs, bt.tracking.use_mlflow)
    variants: list[Variant] = []
    block = event_block(p, cfg.risk.event_window)
    runs: tuple[tuple[str, pd.Series | None], ...] = (
        (("portfolio_no_regime", None), ("portfolio_regime", reg["mult"]))
        if cfg.risk.portfolio.use_regime
        else (("portfolio", None),)  # D33: no market filter; the score decides
    )
    for name, mult in runs:
        res = run_portfolio(p, scores, gate, cfg, regime_mult=mult, start=start, entry_block=block)
        st = run_portfolio(
            p,
            scores,
            gate,
            cfg,
            regime_mult=mult,
            start=start,
            cost_multiplier=bt.execution.cost_stress_multiplier,
            entry_block=block,
        )
        full = metrics.portfolio_metrics(res.daily, p.bench, bt.risk_free_annual)
        oos = metrics.portfolio_metrics(_slice(res.daily, oos_start), p.bench, bt.risk_free_annual)
        oos_s = metrics.portfolio_metrics(_slice(st.daily, oos_start), p.bench, bt.risk_free_annual)
        eq = res.daily["equity"]
        by_year = eq.groupby(pd.DatetimeIndex(eq.index).year).agg(
            lambda s: s.iloc[-1] / s.iloc[0] - 1
        )
        sr, sk, ku, t = metrics.returns_moments(_slice(res.daily, oos_start))
        v = Variant(name, res, st, full, oos, oos_s, by_year, _turnover(res), float("nan"))
        v.run_id = tracker.log_run(
            name,
            {
                "portfolio": cfg.risk.portfolio.model_dump(),
                "regime": cfg.risk.regime.model_dump(),
                "sign_aware": cfg.factors.ic.combine.sign_aware,
                "oos_start": str(oos_start.date()),
                "nse": {
                    "delisted_included": p.delisted_last is not None,
                    "dividends": bool(bt.execution.credit_dividends and p.div_yield is not None),
                    "event_block": block is not None,
                    "sectors": p.sector is not None,
                },
            },
            metrics.nan_to_none(
                {
                    **{f"oos_{k}": val for k, val in oos.items()},
                    **oos,
                    "sharpe_pp": sr,
                    "turnover": v.turnover,
                }
            ),
            tags={"kind": "portfolio"},
            artifacts={"daily": res.daily, "trades": res.trades},
        )
        n, var = tracker.trial_stats()
        v.dsr = dsr.deflated_sharpe(sr, sk, ku, t, max(n, 1), var)
        vc = bt.validation
        v.checks = {
            f"OOS Sharpe >= {cfg.sleeves.validation.sharpe_min_oos}": oos.get("sharpe", -9)
            >= cfg.sleeves.validation.sharpe_min_oos,
            "OOS max DD <= 15%": oos.get("max_dd", -1) >= -0.15,
            "2x costs: OOS excess return > 0": oos_s.get("cagr", -1) > bt.risk_free_annual,
            f"DSR > {vc.dsr_pass}": v.dsr > vc.dsr_pass,
        }
        variants.append(v)
    bench_daily = pd.DataFrame({"equity": p.bench, "invested": p.bench, "positions": 1}).dropna()
    bench = metrics.portfolio_metrics(_slice(bench_daily, oos_start), None, bt.risk_free_annual)
    bench["oos_start"] = float(oos_start.year)
    return variants, bench, reg
