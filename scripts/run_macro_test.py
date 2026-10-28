"""D39: does cross-asset information improve the D37 trade engine? One run, one table.

Variants (all start from D37, pre-registered 2026-09-26):
  D37  as is (score without the macro factors)
  M1   score WITH the macro spill-over factors (they enter only if statistically significant)
  M2   correlation risk: skip a buy whose average 60-day correlation with the holdings > 0.6;
       half risk on new buys during a market-wide correlation spike (> 90th percentile)
  M3   protect open profits: if today's market + sector moves imply a hit > 1 daily σ to a
       holding that is in profit, tighten its trailing stop to highest close − 1σ₁₀
  ALL  M1 + M2 + M3
Stability check: D37 and ALL again with SL 1.25σ / 1.75σ and time exit 15 / 25 sessions.
An idea is only worth keeping if it helps in most of the nudged settings, not in one run.

Needs ``macro_backfill.py`` first. Usage::

    python scripts/run_macro_test.py              # everything, stability check on ALL
    python scripts/run_macro_test.py --check M3   # D37 vs M3 only, 7 settings
"""

from __future__ import annotations

import sys
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd

from quantswing.alpha.ic import summarise
from quantswing.alpha.research import AUX, COMPOSITE, composite_variant, edge_ic_series
from quantswing.backtest import metrics
from quantswing.backtest.tracking import RunTracker
from quantswing.core.config import AppConfig, get_config
from quantswing.core.logging import setup_logging
from quantswing.factors.macro import exposures
from quantswing.portfolio.swing import SwingResult, run_swing
from quantswing.portfolio.swing_report import summarise as swing_summary
from quantswing.portfolio.swing_report import write_csvs, write_html
from quantswing.store.panels import FactorStore, load_panels
from quantswing.store.repositories import DataPaths, write_parquet_frame

MAIN = ("D37", "M1", "M2", "M3", "ALL")
NUDGES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("SL 1.25σ", {"sl_sigma": 1.25}),
    ("SL 1.75σ", {"sl_sigma": 1.75}),
    ("hold 15", {"max_hold": 15}),
    ("hold 25", {"max_hold": 25}),
)
EXTRA: tuple[tuple[str, dict[str, Any]], ...] = (  # --check adds these
    ("10 stocks", {"max_positions": 10}),
    ("15 stocks", {"max_positions": 15}),
)


def with_swing(cfg: AppConfig, **kw: Any) -> AppConfig:
    """Copy of ``cfg`` with ``risk.swing`` fields replaced."""
    sw = cfg.risk.swing.model_copy(update=kw)
    return cfg.model_copy(update={"risk": cfg.risk.model_copy(update={"swing": sw})})


def variant(cfg: AppConfig, name: str) -> AppConfig:
    """Pre-registered settings of each variant (the score is chosen separately)."""
    kw: dict[str, Any] = {}
    if name in ("M2", "ALL"):
        kw |= {"corr_book_max": 0.6, "corr_window": 60, "spike_risk_mult": 0.5}
    if name in ("M3", "ALL"):
        kw |= {"macro_protect": True, "macro_protect_k": 1.0, "macro_tsl_sigma": 1.0}
    return with_swing(cfg, **kw) if kw else cfg


def stats(res: SwingResult, bench: pd.Series, rf: float) -> dict[str, float]:
    """Headline numbers of one run."""
    d = res.daily
    m = metrics.portfolio_metrics(d, bench, rf)
    t = res.trades
    done = t[t["exit_reason"] != "Open"] if not t.empty else t
    return {
        "end_L": float(d["equity"].iloc[-1] / 1e5),
        "cagr": m.get("cagr", np.nan),
        "max_dd": m.get("max_dd", np.nan),
        "sharpe": m.get("sharpe", np.nan),
        "trades": float(len(done)),
        "win": float((done["pnl"] > 0).mean()) if len(done) else np.nan,
        "macro_tsl": float((done["exit_reason"] == "MacroTSL").sum()) if len(done) else 0.0,
    }


def yearly(eq: pd.Series) -> pd.Series:
    """Calendar-year returns."""
    y = eq.groupby(pd.DatetimeIndex(eq.index).year)
    first = eq.shift(1).groupby(pd.DatetimeIndex(eq.index).year).first().fillna(y.first())
    out: pd.Series = y.last() / first - 1
    return out


def main() -> int:
    """Run everything and print the tables."""
    warnings.simplefilter("ignore", RuntimeWarning)
    setup_logging()
    t0 = time.monotonic()
    chk = sys.argv[sys.argv.index("--check") + 1] if "--check" in sys.argv else "ALL"
    if chk not in MAIN or chk == "D37":
        print(f"[STOP] --check must be one of {', '.join(MAIN[1:])}")
        return 1
    mains = ("D37", chk) if "--check" in sys.argv else MAIN
    nudges = NUDGES + EXTRA if "--check" in sys.argv else NUDGES
    cfg = get_config()
    paths = DataPaths(cfg)
    store = FactorStore(paths.factors)
    macro_names = {n for n in store.names("neutral") if n.startswith("MAC_")}
    if not macro_names:
        print("[STOP] No macro factors yet - run: python scripts\\macro_backfill.py")
        return 1
    print("Loading data...", flush=True)
    p = load_panels(paths, cfg)
    if p.macro is not None:
        print(f"Drivers loaded: {', '.join(p.macro.columns)}")
    gate = store.read(AUX, "gate").reindex_like(p.close).fillna(0).astype(bool)
    comp20 = store.read("composite", COMPOSITE)
    print("Building the D37 score (without macro factors)...", flush=True)
    c0, ic0 = composite_variant(paths, cfg, 10, 2.0, "COMPOSITE_H10", exclude=macro_names)
    print("Building the M1 score (with macro factors)...", flush=True)
    f_ics: dict[str, pd.Series] = {}
    c1, ic1 = composite_variant(paths, cfg, 10, 2.0, "COMPOSITE_H10M", factor_ics=f_ics)
    print("\nMacro factors (10-day accuracy; |t| >= 2 needed to enter the score):")
    for n in sorted(macro_names):
        fs = summarise(f_ics[n], 10) if n in f_ics else None
        if fs is not None:
            print(
                f"  {n:<12} IC {fs.mean * 100:+.2f}   t {fs.t_nonoverlap:+.2f}   {'USED' if abs(fs.t_nonoverlap) >= 2 else 'not significant'}"
            )
    print("Measuring market / sector sensitivities per stock...", flush=True)
    ex = exposures(p, cfg.factors.macro)
    bt = cfg.backtest
    first = comp20.dropna(how="all").index.min()
    oos = max(
        pd.Timestamp(bt.start), pd.Timestamp(f"{first.year + bt.validation.train_years}-01-01")
    )
    scores = {"D37": (c0.reindex_like(p.close), edge_ic_series(ic0, 10))}
    scores["M1"] = (c1.reindex_like(p.close), edge_ic_series(ic1, 10))
    tracker = RunTracker(paths.runs, bt.tracking.use_mlflow)
    rows: list[dict[str, Any]] = []
    years: dict[str, pd.Series] = {}

    def run(name: str, vc: AppConfig, stress: bool = False) -> SwingResult:
        sc, edge = scores["M1" if name in ("M1", "ALL") else "D37"]
        return run_swing(
            p,
            sc,
            gate,
            vc,
            start=oos,
            edge_ic=edge,
            macro=ex,
            cost_multiplier=bt.execution.cost_stress_multiplier if stress else 1.0,
        )

    for name in mains:
        print(f"  {name} ...", flush=True)
        vc = variant(cfg, name)
        res = run(name, vc)
        st = run(name, vc, stress=True)
        r = {"name": name, "setting": "base", **stats(res, p.bench, bt.risk_free_annual)}
        r["cagr_2x"] = stats(st, p.bench, bt.risk_free_annual)["cagr"]
        rows.append(r)
        years[name] = yearly(res.daily["equity"])
        out = paths.reports / "macro" / name
        sm = swing_summary(res, st, p.bench, bt.risk_free_annual, float(cfg.risk.paper_capital_inr))
        write_csvs(res, sm, out)
        write_html(sm, out, {"rules": [f"D39 variant {name} (see scripts/run_macro_test.py)"]})
        tracker.log_run(
            f"macro_{name}",
            {
                "swing": vc.risk.swing.model_dump(),
                "score": "H10+MAC" if name in ("M1", "ALL") else "H10",
            },
            metrics.nan_to_none(
                dict(metrics.portfolio_metrics(res.daily, p.bench, bt.risk_free_annual))
            ),
            tags={"kind": "variant"},
            artifacts={"daily": res.daily, "trades": res.trades},
        )
    for label, kw in nudges:
        for name in ("D37", chk):
            print(f"  {name} / {label} ...", flush=True)
            res = run(name, with_swing(variant(cfg, name), **kw))
            rows.append(
                {"name": name, "setting": label, **stats(res, p.bench, bt.risk_free_annual)}
            )
    df = pd.DataFrame(rows)
    write_parquet_frame(df, paths.reports / "macro_test.parquet")
    bq = p.bench.dropna().loc[oos:]
    bm = metrics.portfolio_metrics(
        pd.DataFrame({"equity": bq, "invested": bq, "positions": 1}), None, bt.risk_free_annual
    )

    base = df[df["setting"] == "base"]
    print(f"\nRESULTS {oos.date()} -> {p.close.index[-1].date()}  (Rs 10 lakh at the start)\n")
    print(
        f"{'':<6}{'Rs 10L ->':>10}{'Return/yr':>11}{'Worst fall':>12}{'Sharpe':>8}{'2x costs':>10}{'Trades':>8}{'Win':>6}{'MacroTSL':>10}"
    )
    for rr in base.to_dict("records"):
        q = pd.Series(rr)
        print(
            f"{q['name']:<6}{q['end_L']:>8.1f} L{q['cagr']:>11.1%}{q['max_dd']:>12.1%}{q['sharpe']:>8.2f}"
            f"{q['cagr_2x']:>10.1%}{q['trades']:>8.0f}{q['win']:>6.0%}{q['macro_tsl']:>10.0f}"
        )
    print(
        f"{'Nifty':<6}{bq.iloc[-1] / bq.iloc[0] * 10:>8.1f} L{bm.get('cagr', np.nan):>11.1%}{bm.get('max_dd', np.nan):>12.1%}{bm.get('sharpe', np.nan):>8.2f}"
    )
    by = yearly(bq)
    print("\nYear   " + "".join(f"{n:>9}" for n in years) + f"{'Nifty':>9}")
    for yv in by.index:
        print(
            f"{yv}   "
            + "".join(f"{years[n].get(yv, np.nan):>+9.1%}" for n in years)
            + f"{by[yv]:>+9.1%}"
        )
    print("\nStability check (return/yr | worst fall):")
    print(f"{'setting':<11}{'D37':>22}{chk:>22}   {chk} better?")
    wins = 0
    settings = ["base", *[n for n, _ in nudges]]
    for s in settings:
        a = df[(df["name"] == "D37") & (df["setting"] == s)].iloc[0]
        b = df[(df["name"] == chk) & (df["setting"] == s)].iloc[0]
        better = (b["cagr"] / abs(b["max_dd"])) > (a["cagr"] / abs(a["max_dd"]))
        wins += int(better)
        print(
            f"{s:<11}{a['cagr']:>12.1%} | {a['max_dd']:>6.1%}{b['cagr']:>12.1%} | {b['max_dd']:>6.1%}"
            f"   {'yes' if better else 'no'}"
        )
    print(f"\n{chk} beats D37 on return / worst fall in {wins} of {len(settings)} settings.")
    print(f"Reports: {paths.reports / 'macro'}\\<variant>\\swing_report.html")
    print(f"({time.monotonic() - t0:.0f}s; every main variant logged to the run tracker)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
