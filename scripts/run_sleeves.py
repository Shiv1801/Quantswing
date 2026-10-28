"""P4 exit test: each sleeve passes walk-forward validation or is marked OFF (07 §1).

Needs P1–P2 outputs (backfill, run_factors, p2_ic_report). Runs S1–S5 (S5 needs NSE data), fits exits and
p(win) on training years only, tests each following year, writes
reports/p4_sleeves_report.html and data/processed/sleeves_fitted.json.

Usage::

    python scripts/run_sleeves.py              # all enabled sleeves
    python scripts/run_sleeves.py --only S2    # one sleeve
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from quantswing.core.config import get_config
from quantswing.core.logging import setup_logging
from quantswing.sleeves.base import MissingFactorError
from quantswing.sleeves.report import write_report
from quantswing.sleeves.runner import fitted_path, run_all
from quantswing.store.repositories import DataPaths

NOTES = [
    "Rules per 03 §4; exits (SL, partial, trail, time stop) fitted from MAE/MFE of training signals only.",
    "Entries ranked by EV = p(win)·avgWin − (1−p)·avgLoss − costs (R units); only EV > 0 is traded.",
    "S1 volume confirmation uses the entry day's full volume as a proxy for projected intraday volume (D27).",
    "Not yet applied: regime filter (P5), S3 pyramid tranche (D27), DELZ (D22). Survivorship bias applies (D22).",
    "Verdict ON needs OOS Sharpe >= 0.7, >= 30 OOS trades and positive expectancy at 2x costs (05 §3).",
]


def main(argv: list[str] | None = None) -> int:
    """Run P4. Returns 0 when every sleeve has a verdict."""
    ap = argparse.ArgumentParser(description="QuantSwing P4: sleeve validation")
    ap.add_argument("--only", help="comma-separated sleeves, e.g. S1,S4")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args(argv)
    setup_logging()
    cfg = get_config()
    paths = DataPaths(cfg)
    if not (paths.factors / "aux").exists():
        print("[STOP] No factors yet - run run_factors.py and p2_ic_report.py first.")
        return 1
    t0 = time.monotonic()
    names = [s.strip().upper() for s in args.only.split(",")] if args.only else None
    print("Validating sleeves (walk-forward, a few minutes)...")
    try:
        res = run_all(paths, cfg, names, progress=lambda m: print(f"  {m}", flush=True))
    except MissingFactorError as e:
        print(f"[STOP] Factors are older than the code: {e}")
        print("       Run: python scripts\\run_factors.py  then  python scripts\\p2_ic_report.py")
        return 1
    out = write_report(res, paths.reports, NOTES)
    print(
        f"\n{'sleeve':<7}{'verdict':<8}{'signals':>9}{'trades':>8}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'ExpR':>7}{'2xExpR':>8}"
    )
    for r in res:
        print(
            f"{r.name:<7}{r.verdict:<8}{r.n_signals:>9,}{r.trade.get('trades', 0):>8.0f}"
            f"{r.port.get('cagr', float('nan')):>8.1%}{r.port.get('sharpe', float('nan')):>8.2f}"
            f"{r.port.get('max_dd', float('nan')):>8.1%}{r.trade.get('expectancy_r', float('nan')):>+7.2f}"
            f"{r.stress.get('expectancy_r', float('nan')):>+8.2f}"
        )
        for why in r.reasons:
            print(f"        - {why}")
    ok = bool(res) and all(r.verdict in ("ON", "OFF") for r in res)
    print(
        f"\nReport: {out}\nLive parameters: {fitted_path(paths)}   ({time.monotonic() - t0:.0f}s)"
    )
    print(
        f"\nP4 EXIT TEST: {'PASS' if ok else 'FAIL'}  (ON: {', '.join(r.name for r in res if r.verdict == 'ON') or 'none'})"
    )
    if sys.platform == "win32" and not args.no_open:
        os.startfile(out)  # type: ignore[attr-defined]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
