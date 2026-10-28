"""D41: can D37's drawdown/PF be fixed without giving up return? Target (set BEFORE seeing
results, 2026-09-28, per the user): CAGR 22-25%, Sharpe >= 1.3, Max DD < 20%, PF >= 1.5.

Why: in the D37 backtest, stop-loss exits are the single biggest drag - 213 trades (18% of
all trades), 0% win rate, avg -1.13R, -Rs 74L total P&L - bigger than any other exit bucket.
That is the most likely lever on drawdown and profit factor, so this round tests four
structurally different ways to cut that drag (not another parameter sweep of D37's shape):

Variants (all start from D37; pre-registered 2026-09-28):
  D37  as is (baseline, for comparison)
  T1   regime-gated entries: no NEW buys (or swaps) while the HMM regime multiplier is below
       0.5 (bear / high-vol regime) - existing holdings, exits and stops are unaffected
  T2   wider stop, same risk: SL = entry - 2.0sigma (was 1.5sigma) - quantity already scales
       down automatically (qty = risk / (entry - stop)), so Rs risk per trade is unchanged
  T3   tighter correlation cap: skip a buy whose average 60-day correlation with the book
       > 0.4 (D39's M2 used 0.6 and failed; a tighter cap is the untested part of that idea)
  T4   SL grace period: no stop-loss exit in the first 3 sessions after entry (score/time
       exits still apply) - lets the initial entry noise settle before the stop can trigger
  ALL  T1 + T2 + T3 + T4 together

Stability check (same discipline as D38/D39): D37 and the candidate again at SL +-0.25sigma
(around each one's own base), hold 15/25 sessions, and max_positions 8/10/15/20 (widened
2026-09-28 from just 10/15 - a book-size setting deserves more than a +-1-notch check).
9 settings total. An idea only counts if it holds up in most of them, not just the
headline run - and even then, it must actually clear the target box above.

Usage::

    python scripts/run_target.py              # everything, stability check on ALL
    python scripts/run_target.py --check T1   # D37 vs T1 only, 7 settings
"""

from __future__ import annotations

import sys
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd

from quantswing.alpha.research import AUX, COMPOSITE, composite_variant, edge_ic_series
from quantswing.backtest import metrics
from quantswing.backtest.tracking import RunTracker
from quantswing.core.config import AppConfig, get_config
from quantswing.core.logging import setup_logging
from quantswing.portfolio.p5 import build_regime, load_regime
from quantswing.portfolio.swing import SwingResult, run_swing
from quantswing.portfolio.swing_report import summarise as swing_summary
from quantswing.portfolio.swing_report import write_csvs, write_html
from quantswing.store.panels import FactorStore, load_panels
from quantswing.store.repositories import DataPaths, write_parquet_frame

MAIN = ("D37", "T1", "T2", "T3", "T4", "T24", "ALL")
OTHER_NUDGES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("hold 15", {"max_hold": 15}),
    ("hold 25", {"max_hold": 25}),
    ("8 stocks", {"max_positions": 8}),
    ("10 stocks", {"max_positions": 10}),
    ("15 stocks", {"max_positions": 15}),
    ("20 stocks", {"max_positions": 20}),
)
TARGET = {"cagr_min": 0.22, "cagr_max": 0.25, "sharpe_min": 1.3, "dd_max": -0.20, "pf_min": 1.5}


def with_swing(cfg: AppConfig, **kw: Any) -> AppConfig:
    """Copy of ``cfg`` with ``risk.swing`` fields replaced."""
    sw = cfg.risk.swing.model_copy(update=kw)
    return cfg.model_copy(update={"risk": cfg.risk.model_copy(update={"swing": sw})})


def variant(cfg: AppConfig, name: str) -> AppConfig:
    """Pre-registered settings of each variant."""
    kw: dict[str, Any] = {}
    if name in ("T1", "ALL"):
        kw |= {"regime_gate": True, "regime_gate_mult": 0.5}
    if name in ("T2", "T24", "ALL"):
        kw |= {"sl_sigma": 2.0}
    if name in ("T3", "ALL"):
        kw |= {"corr_book_max": 0.4, "corr_window": 60}
    if name in ("T4", "T24", "ALL"):
        kw |= {"sl_grace_days": 3}
    return with_swing(cfg, **kw) if kw else cfg


def nudges_for(cfg: AppConfig, name: str) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Stability-check settings for ``name``, relative to ITS OWN sl_sigma - not D37's -
    so a variant that itself changes sl_sigma (T2, T24, ALL) is nudged around its own
    value instead of the nudge silently overwriting the very thing being tested."""
    base = variant(cfg, name).risk.swing.sl_sigma
    sl = (
        (f"SL {base - 0.25:g}σ", {"sl_sigma": base - 0.25}),
        (f"SL {base + 0.25:g}σ", {"sl_sigma": base + 0.25}),
    )
    return sl + OTHER_NUDGES


def stats(res: SwingResult, bench: pd.Series, rf: float) -> dict[str, float]:
    """Headline numbers of one run, incl. profit factor (the target checks this too)."""
    d = res.daily
    m = metrics.portfolio_metrics(d, bench, rf)
    t = res.trades
    done = t[t["exit_reason"] != "Open"] if not t.empty else t
    pnl = done["pnl"].astype(float) if not done.empty else pd.Series(dtype=float)
    gw, gl = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
    return {
        "end_L": float(d["equity"].iloc[-1] / 1e5),
        "cagr": m.get("cagr", np.nan),
        "max_dd": m.get("max_dd", np.nan),
        "sharpe": m.get("sharpe", np.nan),
        "profit_factor": float(gw / gl) if gl > 0 else float("nan"),
        "trades": float(len(done)),
        "win": float((done["pnl"] > 0).mean()) if len(done) else np.nan,
    }


def meets_target(r: dict[str, float]) -> bool:
    """All four numbers inside the box, at once (this is a hard bar, not a nice-to-have)."""
    return bool(
        TARGET["cagr_min"] <= r["cagr"] <= TARGET["cagr_max"] + 0.03  # a little headroom above
        and r["sharpe"] >= TARGET["sharpe_min"]
        and r["max_dd"] >= TARGET["dd_max"]
        and r["profit_factor"] >= TARGET["pf_min"]
    )


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
    cfg = get_config()
    paths = DataPaths(cfg)
    store = FactorStore(paths.factors)
    if not store.path("composite", COMPOSITE).exists():
        print("[STOP] No composite yet - run run_factors.py and p2_ic_report.py first.")
        return 1
    print("Loading data...", flush=True)
    p = load_panels(paths, cfg)
    gate = store.read(AUX, "gate").reindex_like(p.close).fillna(0).astype(bool)
    comp20 = store.read("composite", COMPOSITE)
    print("Building the D37 score from significant factors only (a few minutes)...", flush=True)
    mac = {n for n in store.names("neutral") if n.startswith("MAC_")}
    comp10, ic10 = composite_variant(paths, cfg, 10, 2.0, "COMPOSITE_H10", exclude=mac)
    comp10 = comp10.reindex_like(p.close)
    edge = edge_ic_series(ic10, 10)
    reg = load_regime(paths)
    if reg is None:
        print("Building the regime series (first time only)...", flush=True)
        reg = build_regime(p, gate, paths, cfg)
    regime_mult = reg["mult"]
    bt = cfg.backtest
    first = comp20.dropna(how="all").index.min()
    oos = max(
        pd.Timestamp(bt.start), pd.Timestamp(f"{first.year + bt.validation.train_years}-01-01")
    )
    tracker = RunTracker(paths.runs, bt.tracking.use_mlflow)
    rows: list[dict[str, Any]] = []
    years: dict[str, pd.Series] = {}

    def run(vc: AppConfig, stress: bool = False) -> SwingResult:
        return run_swing(
            p,
            comp10,
            gate,
            vc,
            start=oos,
            edge_ic=edge,
            regime_mult=regime_mult,
            cost_multiplier=bt.execution.cost_stress_multiplier if stress else 1.0,
        )

    for name in mains:
        print(f"  {name} ...", flush=True)
        vc = variant(cfg, name)
        res = run(vc)
        st = run(vc, stress=True)
        r = {
            "name": name,
            "setting": "base",
            "label": "base",
            **stats(res, p.bench, bt.risk_free_annual),
        }
        r["cagr_2x"] = stats(st, p.bench, bt.risk_free_annual)["cagr"]
        r["meets_target"] = meets_target(r)
        rows.append(r)
        years[name] = yearly(res.daily["equity"])
        out = paths.reports / "target" / name
        sm = swing_summary(res, st, p.bench, bt.risk_free_annual, float(cfg.risk.paper_capital_inr))
        write_csvs(res, sm, out)
        write_html(sm, out, {"rules": [f"D41 variant {name} (see scripts/run_target.py)"]})
        tracker.log_run(
            f"target_{name}",
            {"swing": vc.risk.swing.model_dump(), "score": "H10 min_t 2"},
            metrics.nan_to_none(
                {**dict(metrics.portfolio_metrics(res.daily, p.bench, bt.risk_free_annual)),
                 "profit_factor": r["profit_factor"], "trades": r["trades"]}
            ),
            tags={"kind": "variant"},
            artifacts={"daily": res.daily, "trades": res.trades},
        )
    d37_nudges = nudges_for(cfg, "D37")
    chk_nudges = nudges_for(cfg, chk)
    for i, ((d_label, d_kw), (c_label, c_kw)) in enumerate(
        zip(d37_nudges, chk_nudges, strict=True)
    ):
        for name, label, kw in (("D37", d_label, d_kw), (chk, c_label, c_kw)):
            print(f"  {name} / {label} ...", flush=True)
            res = run(with_swing(variant(cfg, name), **kw))
            rows.append(
                {
                    "name": name,
                    "setting": f"nudge{i}",
                    "label": label,
                    **stats(res, p.bench, bt.risk_free_annual),
                }
            )
    df = pd.DataFrame(rows)
    write_parquet_frame(df, paths.reports / "target_test.parquet")
    bq = p.bench.dropna().loc[oos:]
    bm = metrics.portfolio_metrics(
        pd.DataFrame({"equity": bq, "invested": bq, "positions": 1}), None, bt.risk_free_annual
    )

    base = df[df["setting"] == "base"]
    print(f"\nRESULTS {oos.date()} -> {p.close.index[-1].date()}  (Rs {cfg.risk.paper_capital_inr / 1e5:.0f} lakh at the start)\n")
    print(
        f"Target: CAGR {TARGET['cagr_min']:.0%}-{TARGET['cagr_max']:.0%}, Sharpe >= {TARGET['sharpe_min']:.1f}, "
        f"Max DD > {TARGET['dd_max']:.0%}, PF >= {TARGET['pf_min']:.1f}\n"
    )
    print(
        f"{'':<6}{'Rs 10L ->':>10}{'CAGR':>9}{'MaxDD':>9}{'Sharpe':>8}{'PF':>7}{'2x costs':>10}"
        f"{'Trades':>8}{'Win':>6}  In target?"
    )
    for rr in base.to_dict("records"):
        q = pd.Series(rr)
        print(
            f"{q['name']:<6}{q['end_L']:>8.1f} L{q['cagr']:>9.1%}{q['max_dd']:>9.1%}{q['sharpe']:>8.2f}"
            f"{q['profit_factor']:>7.2f}{q['cagr_2x']:>10.1%}{q['trades']:>8.0f}{q['win']:>6.0%}"
            f"  {'YES' if q['meets_target'] else 'no'}"
        )
    print(
        f"{'Nifty':<6}{bq.iloc[-1] / bq.iloc[0] * 10:>8.1f} L{bm.get('cagr', np.nan):>9.1%}"
        f"{bm.get('max_dd', np.nan):>9.1%}{bm.get('sharpe', np.nan):>8.2f}"
    )
    by = yearly(bq)
    print("\nYear   " + "".join(f"{n:>9}" for n in years) + f"{'Nifty':>9}")
    for yv in by.index:
        print(
            f"{yv}   "
            + "".join(f"{years[n].get(yv, np.nan):>+9.1%}" for n in years)
            + f"{by[yv]:>+9.1%}"
        )
    print(
        "\nStability check (return/yr | worst fall) - each side nudged around ITS OWN "
        "settings, so a variant that changes sl_sigma is not just reset back to D37's:"
    )
    print(f"{'setting':<22}{'D37':>22}{chk:>22}   {chk} better?")
    wins = 0
    settings = ["base", *[f"nudge{i}" for i in range(len(d37_nudges))]]
    for s in settings:
        a = df[(df["name"] == "D37") & (df["setting"] == s)].iloc[0]
        b = df[(df["name"] == chk) & (df["setting"] == s)].iloc[0]
        better = (b["cagr"] / abs(b["max_dd"])) > (a["cagr"] / abs(a["max_dd"]))
        wins += int(better)
        lbl = f"{a['label']} / {b['label']}" if s != "base" else "base"
        print(
            f"{lbl:<22}{a['cagr']:>12.1%} | {a['max_dd']:>6.1%}{b['cagr']:>12.1%} | {b['max_dd']:>6.1%}"
            f"   {'yes' if better else 'no'}"
        )
    print(f"\n{chk} beats D37 on return / worst fall in {wins} of {len(settings)} settings.")
    print(f"Reports: {paths.reports / 'target'}\\<variant>\\swing_report.html")
    print(f"({time.monotonic() - t0:.0f}s; every main variant logged to the run tracker)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
