"""P1 exit test (07 §1, adapted for D22): full-market Fyers daily dataset + checks pass.

Reads only local files (no Fyers calls). Prints metrics, the split/bonus adjustment probe,
and ``P1 EXIT TEST: PASS`` / ``FAIL``.

Usage::

    python scripts/p1_exit_test.py
"""

from __future__ import annotations

import sys

from quantswing.core.calendar import TradingCalendar
from quantswing.core.config import get_config
from quantswing.core.logging import setup_logging
from quantswing.ingest.fyers_history import latest_completed_session
from quantswing.ingest.pipeline import exit_report
from quantswing.store.repositories import DataPaths


def main() -> int:
    """Print the P1 report. Returns 0 on PASS."""
    setup_logging()
    cfg = get_config()
    paths = DataPaths(cfg)
    latest = latest_completed_session(TradingCalendar.from_config())
    r = exit_report(paths, latest, cfg)

    print("QuantSwing P1 exit test (data source: Fyers only)\n")
    rows = [
        ("Universe (NSE EQ/BE, current)", f"{r.universe}"),
        ("Symbols with data", f"{r.with_data}  ({r.coverage:.1%})"),
        ("Total daily rows", f"{r.rows:,}"),
        ("Earliest date returned by Fyers", f"{r.earliest}"),
        ("Calendar sessions", f"{r.sessions}  (from {r.calendar_first})"),
        ("Latest completed session", f"{r.latest_session}"),
        ("Symbols up to date", f"{r.fresh_frac:.1%}"),
        ("Rejected rows (bad OHLC)", f"{r.rejected_rows}  ({r.reject_frac:.4%})"),
        ("Quarantined rows (spikes/suspect gaps)", f"{r.quarantined_rows}"),
        ("Split/bonus gaps auto-adjusted", f"{r.adjustments}"),
    ]
    w = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"  {k:<{w}}  {v}")

    print("\n  First-candle year (how far back Fyers goes, per symbol):")
    print("   " + "  ".join(f"{y}:{n}" for y, n in r.earliest_by_year.items()))

    print(f"\n  Issues by check: {r.issue_counts}")

    print("\n  Split/bonus adjustment probe:")
    for p in r.probes:
        obs = "n/a" if p.observed_ratio is None else f"{p.observed_ratio:.3f}"
        print(
            f"   {p.symbol} ex {p.ex_date}: close ratio {obs} (unadjusted would be ~{p.expected_ratio}) -> {p.verdict}"
        )
    verdicts = {p.verdict for p in r.probes}
    ca = r.issue_counts.get("ca_suspect", 0)
    if "UNADJUSTED" in verdicts or r.adjustments > 0:
        print(f"   Fyers history is UNADJUSTED in places: {r.adjustments} gaps auto-adjusted")
        print("   (adj_* columns in prices_daily). Review: data/processed/adjustments.parquet")
    elif verdicts == {"ADJUSTED"}:
        print("   OK: Fyers history is split/bonus-adjusted.")
    if ca:
        print(f"   ! {ca} gaps did not match a known ratio: quarantined for review.")

    print("\n  Checks:")
    for name, ok in r.checks.items():
        print(f"   [{'OK' if ok else 'XX'}] {name}")
    print(f"\nP1 EXIT TEST: {'PASS' if r.passed else 'FAIL'}")
    return 0 if r.passed else 1


if __name__ == "__main__":
    sys.exit(main())
