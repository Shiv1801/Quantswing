"""T24 (locked 2026-09-28, replaces D37): backtest the trade engine (score entry,
SL / half-target / TSL, score & time exits) and write the full results vs Nifty 500.
Same engine as D37; only risk.swing.sl_sigma (2.0, was 1.5) and sl_grace_days (3, was 0)
changed — see config/risk.yaml and 07_ROADMAP_DECISIONS.md (D41) for why.

Outputs (folder ``reports/swing``):
  swing_report.html   charts + tables + every trade (open in a browser)
  swing_trades.csv    one row per trade (open in Excel)
  swing_fills.csv     every buy / sell fill
  swing_daily.csv     daily equity, cash, positions, drawdown vs Nifty 500
  swing_yearly.csv / swing_monthly.csv

Usage::

    python scripts/run_swing.py
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd

from quantswing.alpha.research import AUX, COMPOSITE, composite_variant, edge_ic_series
from quantswing.backtest import dsr, metrics
from quantswing.backtest.tracking import RunTracker
from quantswing.core.config import SwingCfg, get_config
from quantswing.core.logging import setup_logging
from quantswing.factors.macro import exposures
from quantswing.portfolio.swing import run_swing
from quantswing.portfolio.swing_report import lakh, summarise, write_csvs, write_html
from quantswing.store.panels import FactorStore, load_panels
from quantswing.store.repositories import DataPaths

H10 = "COMPOSITE_H10"


def rules(s: SwingCfg) -> list[str]:
    """The rules in plain words (printed and shown in the report)."""
    t = ", ".join(f"z ≥ {a:g} → {b:.2%}" for a, b in s.risk_tiers)
    return [
        "Stocks: the 1,000 most traded (60-session median ₹ value), not under surveillance.",
        "Score: 10-day composite of statistically significant factors, as a z-score across stocks.",
        f"Buy (next open) when z ≥ {s.z_entry:g} and expected gain IC·σ₁₀·z beats "
        f"{s.entry_hurdle_mult:g}× the round-trip cost; skip if it opens > {s.gap_skip_sigma:g}σ above "
        "yesterday's close.",
        f"Size: quantity = risk ₹ ÷ (buy − SL); risk per trade {t} of capital; "
        f"max {s.max_weight:.0%} of capital per stock, max {s.max_positions} stocks, "
        f"total open risk ≤ {s.max_open_risk:.0%}.",
        f"SL = buy − {s.sl_sigma:g}σ₁₀ (resting SL-M order).",
        f"Target = buy + {s.target_sigma:g}σ₁₀: sell {s.target_sell_frac:.0%}, move SL to the buy price.",
        f"TSL: once up {s.tsl_start_sigma:g}σ₁₀, stop = highest close − {s.tsl_sigma:g}σ₁₀.",
        f"Score exit when z < {s.z_exit:g}; time exit after {s.max_hold} sessions; "
        f"no re-buy for {s.cooldown} sessions after a sale.",
        f"Swap the weakest holding only if a new stock's gain beats it by {s.replace_hurdle_mult:g}× cost "
        f"(max {s.max_replacements} per day).",
        "σ₁₀ = the stock's daily volatility (60 sessions) × √10, fixed at the buy.",
    ]


def main() -> int:
    """Run the backtest and print the summary."""
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
    comp20 = store.read("composite", COMPOSITE)
    print("Building the 10-day score from significant factors only (a few minutes)...", flush=True)
    mac = (
        set()
        if cfg.risk.swing.use_macro_factors
        else {n for n in store.names("neutral") if n.startswith("MAC_")}
    )
    comp10, ic10 = composite_variant(paths, cfg, 10, 2.0, H10, exclude=mac)
    comp10 = comp10.reindex_like(p.close)
    edge = edge_ic_series(ic10, 10)
    bt = cfg.backtest
    first = comp20.dropna(how="all").index.min()
    oos = max(
        pd.Timestamp(bt.start), pd.Timestamp(f"{first.year + bt.validation.train_years}-01-01")
    )
    print(
        f"Trading from {oos.date()} with Rs {cfg.risk.paper_capital_inr / 1e5:.0f} lakh...",
        flush=True,
    )
    ex = exposures(p, cfg.factors.macro) if cfg.risk.swing.macro_protect else None
    res = run_swing(p, comp10, gate, cfg, start=oos, edge_ic=edge, macro=ex)
    print("Same again at 2x costs (stress test)...", flush=True)
    stress = run_swing(
        p,
        comp10,
        gate,
        cfg,
        start=oos,
        edge_ic=edge,
        cost_multiplier=bt.execution.cost_stress_multiplier,
        macro=ex,
    )
    if res.daily.empty:
        print("[STOP] No sessions to simulate.")
        return 1
    cap = float(cfg.risk.paper_capital_inr)
    s = summarise(res, stress, p.bench, bt.risk_free_annual, cap)
    out = paths.reports / "swing"
    files = write_csvs(res, s, out)
    rl = rules(cfg.risk.swing)
    page = write_html(s, out, {"rules": rl})
    tracker = RunTracker(paths.runs, bt.tracking.use_mlflow)
    sr, sk, ku, n = metrics.returns_moments(res.daily)
    trade_stats = {**s.stats, "expectancy_r": s.stats["expectancy_R"]}
    tracker.log_run(
        "swing_T24",
        {"swing": cfg.risk.swing.model_dump(), "score": "H10 min_t 2"},
        metrics.nan_to_none({**s.sys, **trade_stats, "sharpe_pp": sr}),
        tags={"kind": "variant"},
        artifacts={"daily": res.daily, "trades": res.trades},
    )
    n_trials, var = tracker.trial_stats()
    dsr_v = dsr.deflated_sharpe(sr, sk, ku, n, max(n_trials, 1), var)

    m, mb, st = s.sys, s.nifty, s.stats
    print(f"\nRESULTS {s.start.date()} -> {s.end.date()}  (Rs {cap / 1e5:.0f} lakh at the start)\n")
    print(f"{'':<28}{'System':>12}{'Nifty 500':>12}")
    for lbl, a, b in (
        ("Money at the end", "Rs " + lakh(s.end_value), "Rs " + lakh(s.nifty_end)),
        ("Return per year", f"{m.get('cagr', np.nan):.1%}", f"{mb.get('cagr', np.nan):.1%}"),
        ("Worst fall", f"{m.get('max_dd', np.nan):.1%}", f"{mb.get('max_dd', np.nan):.1%}"),
        ("Sharpe", f"{m.get('sharpe', np.nan):.2f}", f"{mb.get('sharpe', np.nan):.2f}"),
        ("Return/yr at 2x costs", f"{s.stress.get('cagr', np.nan):.1%}", ""),
        ("Worst fall at 2x costs", f"{s.stress.get('max_dd', np.nan):.1%}", ""),
    ):
        print(f"{lbl:<28}{a:>12}{b:>12}")
    print(
        f"\nTrades {st['trades']:.0f} ({st['trades_per_year']:.0f}/yr) | win rate {st['win_rate']:.0%}"
        f" | avg win {st['avg_win']:+.1%} | avg loss {st['avg_loss']:+.1%}"
        f" | profit factor {st['profit_factor']:.2f} | avg {st['expectancy_R']:+.2f}R"
        f" | avg hold {st['avg_sessions']:.1f} sessions"
    )
    print(
        f"Avg stocks held {st['avg_positions']:.1f} | money invested {st['invested_share']:.0%}"
        f" | costs {st['costs_pa']:.2%}/yr | beta {m.get('beta', np.nan):.2f}"
        f" | alpha {m.get('alpha_ann', np.nan):+.1%}/yr | DSR {dsr_v:.2f}"
    )
    print(
        f"\n{'Year':<6}{'System':>9}{'Nifty':>9}{'Diff':>9}{'Sys fall':>10}{'Trades':>8}{'Win':>6}"
    )
    for y, r in s.yearly.iterrows():
        print(
            f"{y:<6}{r.system:>+9.1%}{r.nifty500:>+9.1%}{r.difference:>+9.1%}"
            f"{r.system_worst_fall:>10.1%}{r.trades:>8.0f}{r.win_rate:>6.0%}"
        )
    print(
        f"\n{'Exit':<10}{'Trades':>8}{'Share':>8}{'Win':>7}{'Avg ret':>9}{'Avg R':>8}{'P&L Rs':>14}"
    )
    for k, r in s.exits.iterrows():
        print(
            f"{k:<10}{r.trades:>8.0f}{r.share:>8.0%}{r.win_rate:>7.0%}{r.avg_return:>+9.1%}"
            f"{r.avg_R:>8.2f}{r.pnl_inr:>14,.0f}"
        )
    print(f"\nReport:    {page}")
    print(f"Trade log: {files['trades']}")
    print(f"({time.monotonic() - t0:.0f}s; logged to the run tracker)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
