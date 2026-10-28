"""P2 step 1: compute all factors, cross-sectionally process them, store the panels.

Needs the P1 database (run backfill.py first). No Fyers calls.

Usage::

    python scripts/run_factors.py --limit 200   # quick trial on 200 stocks
    python scripts/run_factors.py               # all stocks (~several minutes)
"""

from __future__ import annotations

import argparse
import sys
import time

from quantswing.alpha.research import build_factors
from quantswing.core.config import get_config
from quantswing.core.logging import setup_logging
from quantswing.store.repositories import DataPaths


def main(argv: list[str] | None = None) -> int:
    """Build factor panels. Returns 0 on success."""
    ap = argparse.ArgumentParser(description="QuantSwing P2: build factor panels")
    ap.add_argument("--limit", type=int, help="only the first N stocks (trial)")
    args = ap.parse_args(argv)
    setup_logging()
    cfg = get_config()
    paths = DataPaths(cfg)
    if not paths.duckdb.exists():
        print("[STOP] No database yet - run: python scripts\\backfill.py")
        return 1
    t0 = time.monotonic()
    print("Building factors (each line = one factor done)...")
    s = build_factors(paths, cfg, args.limit, progress=lambda n: print(f"  done {n}", flush=True))
    print(
        f"\nSessions {s.sessions}, stocks {s.symbols}, avg stocks passing gate per day {s.gated_avg:.0f}"
    )
    print("Avg names/day with a score: " + ", ".join(f"{k} {v:.0f}" for k, v in s.coverage.items()))
    print(f"Finished in {time.monotonic() - t0:.0f}s. Next: python scripts\\p2_ic_report.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
