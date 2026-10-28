"""Run every pre-registered portfolio variant in one go and summarise in plain terms.

Variants (fixed BEFORE seeing results, 2026-09-26):
  V0  D33 as today (baseline)
  V1  V0 + leak fixes: no drawdown brake (pause new buys only at −30%), no resizing of
      small drifts (±25% of a slot), no results-window block
  V2  V1 + v2 engine: score at a 10-day horizon from significant factors only (|t| ≥ 2),
      conviction sizing (w ∝ score/σ, 5–20% each, up to 15 names, fully invested),
      switch only when the expected gain beats the round-trip cost
  V3  V2 + trailing stop (exit below peak × (1 − 2.5σ√10), 20-session re-entry cooldown)

Final round (D35, fixed BEFORE seeing results, 2026-09-26) — goal: less drawdown, high return:
  F1  V2 + trade less: decide every 10 sessions, swap only if the gain beats 2× the cost
  F2  F1 + spread risk: up to 20 names, max 30% per NSE sector, portfolio risk capped at
      20%/yr (covariance forecast; the rest in cash)
  F3  F2 + trailing stop (as V3)
  F4  F3 + take profit: sell once a stock is up 20% from entry (20-session cooldown)

D36 round (2026-09-26, user: "no reference, no winner rules, just test"):
  B1  F1 + the score sets exposure: invested share = clip(t / 2, 0, 1), where t is the
      t-stat of the score's own daily IC over the last 63 sessions (only ICs whose 10-day
      window has closed — no look-ahead)
  B2  F1 + trailing stop (as F3)
  B3  B1 + trailing stop
Plus a diagnosis table: each factor's mean IC by year (not a trial, changes nothing).

No winner is picked here. Every variant is logged to the run tracker, so the Deflated
Sharpe counts every trial. Default: B1–B3; ``--all`` also re-runs V0–V3 and F1–F4.

Usage::

    python scripts/run_variants.py
"""

from __future__ import annotations

import sys
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd

from quantswing.alpha.events import event_block
from quantswing.alpha.ic import ic_by_year
from quantswing.alpha.research import (
    AUX,
    COMPOSITE,
    composite_variant,
    edge_ic_series,
    ic_exposure_series,
)
from quantswing.backtest import dsr, metrics
from quantswing.backtest.tracking import RunTracker
from quantswing.core.config import AppConfig, get_config
from quantswing.core.logging import setup_logging
from quantswing.portfolio.construction import PortfolioResult, run_portfolio
from quantswing.store.panels import FactorStore, load_panels
from quantswing.store.repositories import DataPaths, write_parquet_frame

H10 = "COMPOSITE_H10"


def variant_cfg(cfg: AppConfig, name: str) -> AppConfig:
    """The pre-registered settings of each variant."""
    pc = cfg.risk.portfolio
    ew = cfg.risk.event_window
    if name == "V0":
        return cfg
    upd: dict[str, Any] = {"dd_brake": False, "halt_dd": 0.30, "rebalance_band": 0.25}
    if name != "V1":
        upd |= {
            "selection": "edge",
            "weighting": "conviction",
            "max_names": 15,
            "min_weight": 0.05,
            "max_weight": 0.20,
            "edge_horizon": 10,
        }
    if name in ("F1", "F2", "F3", "F4", "B1", "B2", "B3"):
        upd |= {"rebalance_every": 10, "switch_cost_mult": 2.0}
    if name in ("F2", "F3", "F4"):
        upd |= {"max_names": 20, "sector_cap": 0.30, "vol_target": True, "vol_target_level": 0.20}
    if name in ("B1", "B3"):
        upd |= {"ic_exposure": True, "ic_exposure_window": 63, "ic_exposure_full_t": 2.0}
    if name in ("V3", "F3", "F4", "B2", "B3"):
        upd |= {"trail_sigma": 2.5, "trail_horizon": 10, "reentry_cooldown": 20}
    if name == "F4":
        upd |= {"take_profit": 0.20}
    risk = cfg.risk.model_copy(
        update={
            "portfolio": pc.model_copy(update=upd),
            "event_window": ew.model_copy(update={"enabled": False}),
        }
    )
    return cfg.model_copy(update={"risk": risk})


def yearly(eq: pd.Series) -> pd.Series:
    """Calendar-year returns."""
    return eq.groupby(pd.DatetimeIndex(eq.index).year).agg(lambda s: s.iloc[-1] / s.iloc[0] - 1)


H10_NAMES = ("V2", "V3", "F1", "F2", "F3", "F4", "B1", "B2", "B3")
ALL = ("V0", "V1", "V2", "V3", "F1", "F2", "F3", "F4", "B1", "B2", "B3")


def cagr_ex(y: pd.Series, drop: int) -> float:
    """Compound yearly return over the calendar years in ``y`` except ``drop``."""
    z = y.drop(drop, errors="ignore").dropna()
    g = float(np.prod(1.0 + z.to_numpy(dtype="float64")))
    return g ** (1 / len(z)) - 1 if len(z) else float("nan")


def main() -> int:
    """Run the variants and print the summary (no winner is picked)."""
    names = ALL if "--all" in sys.argv else ("B1", "B2", "B3")
    warnings.simplefilter("ignore", RuntimeWarning)
    setup_logging()
    t0 = time.monotonic()
    cfg = get_config()
    paths = DataPaths(cfg)
    store = FactorStore(paths.factors)
    if not store.path("composite", COMPOSITE).exists():
        print("[STOP] No composite yet - run run_factors.py and p2_ic_report.py first.")
        return 1
    print("Loading data...", flush=True)
    p = load_panels(paths, cfg)
    gate = store.read(AUX, "gate").reindex_like(p.close).fillna(0).astype(bool)
    comp20 = store.read("composite", COMPOSITE).reindex_like(p.close)
    print("Building the 10-day score from significant factors only (a few minutes)...", flush=True)
    f_ics: dict[str, pd.Series] = {}
    comp10, ic10 = composite_variant(paths, cfg, 10, 2.0, H10, factor_ics=f_ics)
    comp10 = comp10.reindex_like(p.close)
    edge = edge_ic_series(ic10, 10)
    bt = cfg.backtest
    first = comp20.dropna(how="all").index.min()
    oos = max(
        pd.Timestamp(bt.start), pd.Timestamp(f"{first.year + bt.validation.train_years}-01-01")
    )
    start = pd.Timestamp(bt.start)
    tracker = RunTracker(paths.runs, bt.tracking.use_mlflow)
    bench = p.bench
    by_bench = yearly(bench[bench.index >= oos].dropna())
    rows: list[dict[str, Any]] = []
    years: dict[str, pd.Series] = {}
    expo_used: pd.Series | None = None
    for name in names:
        vc = variant_cfg(cfg, name)
        vp = vc.risk.portfolio
        scores = comp10 if name in H10_NAMES else comp20
        blk = event_block(p, vc.risk.event_window)
        print(f"  {name} ...", flush=True)
        kw: dict[str, Any] = dict(start=start, entry_block=blk)
        if name in H10_NAMES:
            kw["edge_ic"] = edge
        if vp.ic_exposure:
            expo_used = ic_exposure_series(ic10, 10, vp.ic_exposure_window, vp.ic_exposure_full_t)
            kw["exposure"] = expo_used
        res: PortfolioResult = run_portfolio(p, scores, gate, vc, **kw)
        st: PortfolioResult = run_portfolio(
            p, scores, gate, vc, cost_multiplier=bt.execution.cost_stress_multiplier, **kw
        )
        d = res.daily[res.daily.index >= oos]
        m = metrics.portfolio_metrics(d, bench, bt.risk_free_annual)
        ms = metrics.portfolio_metrics(st.daily[st.daily.index >= oos], bench, bt.risk_free_annual)
        t = res.trades
        t = t[pd.to_datetime(t["date"]) >= oos] if not t.empty else t
        yrs = len(d) / 252
        sr, sk, ku, n = metrics.returns_moments(d)
        run_id = tracker.log_run(
            f"variant_{name}",
            {"portfolio": vp.model_dump(), "score": "H10" if name in H10_NAMES else "H20"},
            metrics.nan_to_none({**m, "sharpe_pp": sr}),
            tags={"kind": "variant"},
            artifacts={"daily": res.daily, "trades": res.trades},
        )
        n_trials, var = tracker.trial_stats()
        y = yearly(d["equity"])
        years[name] = y
        dd = m.get("max_dd", np.nan)
        rows.append(
            {
                "name": name,
                "run_id": run_id,
                "cagr": m.get("cagr", np.nan),
                "cagr_ex2020": cagr_ex(y, 2020),
                "sharpe": m.get("sharpe", np.nan),
                "max_dd": dd,
                "cagr_2x": ms.get("cagr", np.nan),
                "invested": float((d["invested"] / d["equity"]).mean()),
                "positions": float(d["positions"].mean()),
                "costs": float(t["cost"].sum() / d["equity"].mean() / yrs) if not t.empty else 0.0,
                "trades_yr": len(t) / yrs,
                "calmar": m.get("cagr", np.nan) / abs(dd) if dd else np.nan,
                "trail_exits": float((t["reason"] == "trail").sum()) if not t.empty else 0.0,
                "beat_years": float((y.reindex(by_bench.index) > by_bench).mean()),
                "end_value": float(d["equity"].iloc[-1] / d["equity"].iloc[0] * 10.0),
                "dsr": dsr.deflated_sharpe(sr, sk, ku, n, max(n_trials, 1), var),
            }
        )
    df = pd.DataFrame(rows)
    bq = bench.dropna().loc[oos:]
    bm = metrics.portfolio_metrics(
        pd.DataFrame({"equity": bq, "invested": bq, "positions": 1}), None, bt.risk_free_annual
    )
    b_end = float(bq.iloc[-1] / bq.iloc[0] * 10.0)
    write_parquet_frame(df, paths.reports / "variants.parquet")

    print(f"\nOut-of-sample from {oos.date()}  (Rs 10 lakh invested at the start)\n")
    print(
        f"{'':<5}{'Rs 10L ->':>10}{'Return/yr':>11}{'ex-2020':>9}{'Sharpe':>8}{'Worst fall':>12}"
        f"{'2x costs':>10}{'Invested':>10}{'Stocks':>8}{'Trades/yr':>11}{'Costs/yr':>10}"
        f"{'Beat Nifty':>12}{'Ret/Fall':>10}{'DSR':>6}"
    )
    for r in df.itertuples():
        print(
            f"{r.name:<5}{r.end_value:>8.1f} L{r.cagr:>11.1%}{r.cagr_ex2020:>9.1%}{r.sharpe:>8.2f}"
            f"{r.max_dd:>12.1%}{r.cagr_2x:>10.1%}{r.invested:>10.0%}{r.positions:>8.1f}"
            f"{r.trades_yr:>11.0f}{r.costs:>10.2%}{r.beat_years:>10.0%} yrs{r.calmar:>10.2f}"
            f"{r.dsr:>6.2f}"
        )
    print(
        f"{'Nifty':<5}{b_end:>8.1f} L{bm.get('cagr', np.nan):>11.1%}"
        f"{cagr_ex(by_bench, 2020):>9.1%}{bm.get('sharpe', np.nan):>8.2f}"
        f"{bm.get('max_dd', np.nan):>12.1%}"
    )
    print("\nYear   " + "".join(f"{n:>9}" for n in years) + f"{'Nifty':>9}")
    for yv in by_bench.index:
        print(
            f"{yv}   "
            + "".join(f"{years[n].get(yv, np.nan):>+9.1%}" for n in years)
            + f"{by_bench[yv]:>+9.1%}"
        )
    if expo_used is not None:  # how much the score let us invest, per year
        e = expo_used[expo_used.index >= oos]
        ey = e.groupby(pd.DatetimeIndex(e.index).year)
        print("\nScore-set exposure (B1/B3):  year  average  min  % of days at 0")
        for yv, g in ey:
            print(f"{'':<29}{yv}  {g.mean():>7.0%}{g.min():>5.0%}{(g <= 0).mean():>10.0%}")
    # ---- diagnosis: which factors worked in which year (not a trial, changes nothing)
    tab = pd.DataFrame({k: ic_by_year(v[v.index >= oos]) for k, v in f_ics.items()}).T
    tab.loc["SCORE (10-day)"] = ic_by_year(ic10[ic10.index >= oos])
    tab = tab.sort_values(by=list(tab.columns[-2:]), ascending=False) if tab.shape[1] >= 2 else tab
    print("\nFactor accuracy by year (mean daily IC x100; + = predicted well, - = backwards)")
    print(f"{'factor':<28}" + "".join(f"{c:>7}" for c in tab.columns))
    for k, row in tab.iterrows():
        print(f"{str(k)[:27]:<28}" + "".join(f"{v * 100:>+7.1f}" for v in row))
    write_parquet_frame(
        tab.reset_index(names="factor"), paths.reports / "factor_ic_by_year.parquet"
    )
    print(
        f"\n({time.monotonic() - t0:.0f}s; every variant logged to the run tracker; no winner picked)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
