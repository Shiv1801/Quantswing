"""P3 exit test: reproducible baseline backtest, logged (07 §1).

Runs the composite-score baseline through the engine at 1x and 2x costs, computes metrics,
Monte Carlo, Deflated Sharpe, year split and triple-barrier label mix, logs the run in
runs/, writes reports/p3_backtest_report.html and prints ``P3 EXIT TEST: PASS`` when the
run is reproducible and logged. Needs P1 (backfill) and P2 (run_factors + p2_ic_report).

Usage::

    python scripts/run_backtest.py
"""

from __future__ import annotations

import os
import sys
import time

from quantswing.backtest.baseline import run_baseline
from quantswing.backtest.report import write_report
from quantswing.core.config import get_config
from quantswing.core.logging import setup_logging
from quantswing.store.panels import load_panels
from quantswing.store.repositories import DataPaths

NOTES = [
    "Baseline = long the highest composite scores (P2 IC-IR composite) with the 04 exit rules; not a tuned sleeve.",
    "Exit multipliers (k_sl, partial, trail, time stop) are defaults until fitted per sleeve from MAE/MFE in P4.",
    "Survivorship bias (D22): only currently listed stocks -> results are optimistic.",
    "Entries at the next open after the signal; stops/targets on daily high/low with worst-case ordering.",
    "Costs: full Fyers delivery schedule + spread/impact slippage; stress run doubles both.",
    "Strategy-quality tests (DSR, Monte Carlo, cost stress, year split) are informational for the baseline; sleeves must pass them in P4.",
]


def main() -> int:
    """Run the baseline. Returns 0 on PASS."""
    setup_logging()
    cfg = get_config()
    paths = DataPaths(cfg)
    if not (paths.factors / "composite").exists():
        print("[STOP] No composite score yet - run run_factors.py and p2_ic_report.py first.")
        return 1
    t0 = time.monotonic()
    print("Running baseline backtest (1x and 2x costs, Monte Carlo, labels)...")
    rep = run_baseline(paths, cfg)
    bench = load_panels(paths, cfg, limit=1).bench
    out = write_report(rep, bench, paths.reports, NOTES)
    p, t = rep.port, rep.trade
    print(
        f"\n  CAGR {p.get('cagr', float('nan')):.1%}   Sharpe {p.get('sharpe', float('nan')):.2f}   "
        f"Max DD {p.get('max_dd', float('nan')):.1%}   Trades {t.get('trades', 0):.0f}"
    )
    print(
        f"  Win rate {t.get('win_rate', float('nan')):.1%}   Expectancy {t.get('expectancy_r', float('nan')):+.2f}R   "
        f"Profit factor {t.get('profit_factor', float('nan')):.2f}   DSR {rep.dsr:.2f}"
    )
    print("\n  Exit criteria:")
    for k, v in rep.checks.items():
        print(f"   [{'OK' if v else 'XX'}] {k}")
    print("  Strategy quality (info for baseline; required per sleeve in P4):")
    for k, v in rep.quality.items():
        print(f"   [{'OK' if v else '--'}] {k}")
    print(f"\n  Run id: {rep.run_id}   Report: {out}   ({time.monotonic() - t0:.0f}s)")
    passed = all(rep.checks.values())
    print(f"\nP3 EXIT TEST: {'PASS' if passed else 'FAIL'}")
    if sys.platform == "win32" and "--no-open" not in sys.argv:
        os.startfile(out)  # type: ignore[attr-defined]
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
