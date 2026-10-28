"""P5: regime model + ranked factor portfolio (D28 composite, D29 portfolio).

Needs P2 outputs (run_factors.py + p2_ic_report.py). Writes
reports/p5_portfolio_report.html and data/processed/regime.parquet, logs each variant.
Prints ``P5 EXIT TEST: PASS`` only if the regime variant meets the out-of-sample bar
(Sharpe >= 0.7, max DD <= 15%, beats cash at 2x costs, DSR > 0.95).

Usage::

    python scripts/run_portfolio.py
"""

from __future__ import annotations

import os
import sys
import time

from quantswing.core.config import get_config
from quantswing.core.logging import setup_logging
from quantswing.portfolio.p5 import run_p5
from quantswing.portfolio.report import write_report
from quantswing.store.panels import load_panels
from quantswing.store.repositories import DataPaths

NOTES = [
    "D33: universe = 1,000 most traded stocks (60-session median value), re-ranked at every decision.",
    "Composite = IC-IR weighted, sector- and size-neutral factors (D28 sign handling, weights from past ICs only).",
    "Book: up to 10 stocks with a positive score, 10% each; keep while within the top 20; empty slots stay in cash. Weekly, at the next open.",
    "No market filter or volatility target; drawdown brake (D26) and results-window entry block (±3 sessions) apply.",
    "Costs: full Fyers delivery schedule + spread/impact slippage; idle cash earns 6.5%; dividends credited; delisted stocks included (mergers at full value, others −30%).",
    "OOS = years after the first 3 years of composite history; nothing else is fitted in-sample.",
]


def main() -> int:
    """Run P5. Returns 0 on PASS."""
    setup_logging()
    cfg = get_config()
    paths = DataPaths(cfg)
    if not (paths.factors / "composite").exists():
        print("[STOP] No composite yet - run run_factors.py and p2_ic_report.py first.")
        return 1
    t0 = time.monotonic()
    print("Building regime model + portfolio variants (a few minutes)...")
    variants, bench, reg = run_p5(paths, cfg)
    bench_px = load_panels(paths, cfg, limit=1).bench
    out = write_report(variants, bench, reg, bench_px, paths.reports, NOTES)
    oos_from = int(bench.get("oos_start", 0))
    print(f"\nOut-of-sample from {oos_from}:")
    print(
        f"{'variant':<22}{'CAGR':>8}{'Vol':>7}{'Sharpe':>8}{'MaxDD':>8}{'Calmar':>8}{'2x CAGR':>9}{'DSR':>6}"
    )
    for v in variants:
        o, s = v.oos, v.oos_stress
        print(
            f"{v.name:<22}{o.get('cagr', float('nan')):>8.1%}{o.get('vol', float('nan')):>7.1%}{o.get('sharpe', float('nan')):>8.2f}"
            f"{o.get('max_dd', float('nan')):>8.1%}{o.get('calmar', float('nan')):>8.2f}{s.get('cagr', float('nan')):>9.1%}{v.dsr:>6.2f}"
        )
    print(
        f"{'Nifty 500 (price)':<22}{bench.get('cagr', float('nan')):>8.1%}{bench.get('vol', float('nan')):>7.1%}{bench.get('sharpe', float('nan')):>8.2f}{bench.get('max_dd', float('nan')):>8.1%}"
    )
    st = reg["state"].dropna()
    if len(st):
        used = "" if cfg.risk.portfolio.use_regime else ", display only"
        print(
            f"\nRegime today: {st.iloc[-1]}  (exposure x{reg['mult'].dropna().iloc[-1]:.1f}{used})"
        )
    best = variants[-1]
    tr = best.res.trades
    if not tr.empty and "reason" in tr:
        dl = tr[tr["reason"] == "delisted"]
        inv = best.res.daily["invested"] / best.res.daily["equity"]
        print(
            f"\nAvg invested {inv.mean():.0%} | positions avg {best.res.daily['positions'].mean():.1f}"
            f" | delisted exits {len(dl)} (Rs {dl['value'].sum():,.0f} recovered)"
        )
    print(f"\nChecks ({best.name}):")
    for k, ok in best.checks.items():
        print(f"   [{'OK' if ok else 'XX'}] {k}")
    passed = all(best.checks.values())
    print(f"\nReport: {out}   ({time.monotonic() - t0:.0f}s)")
    print(f"\nP5 EXIT TEST: {'PASS' if passed else 'FAIL'}")
    if sys.platform == "win32" and "--no-open" not in sys.argv:
        os.startfile(out)  # type: ignore[attr-defined]
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
