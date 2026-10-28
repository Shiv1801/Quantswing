"""P1 backfill: all NSE equities' daily history from Fyers (D22), resumable.

Steps: universe -> indices -> calendar -> equities -> validation -> DuckDB -> exit summary.
Safe to stop (Ctrl+C) and re-run: finished symbols are skipped.

Usage::

    python scripts/backfill.py --limit 20     # quick trial on 20 symbols (~2 min)
    python scripts/backfill.py                # everything (~3 hours first time)
    python scripts/backfill.py --symbols NSE:TCS-EQ,NSE:INFY-EQ
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date

from quantswing.broker.session import open_data_client
from quantswing.core.calendar import TradingCalendar
from quantswing.core.config import get_config
from quantswing.core.logging import get_logger, setup_logging
from quantswing.core.types import TokenMissingError
from quantswing.ingest import pipeline
from quantswing.ingest.fyers_history import RunSummary, latest_completed_session
from quantswing.store.database import build_database
from quantswing.store.repositories import DataPaths

log = get_logger("scripts.backfill")


def fmt_dur(sec: float) -> str:
    """Seconds -> '1h05m' / '4m10s'."""
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def main(argv: list[str] | None = None) -> int:
    """Run the P1 pipeline. Returns 0 on success."""
    ap = argparse.ArgumentParser(description="QuantSwing P1 backfill (Fyers daily history)")
    ap.add_argument("--limit", type=int, help="only the first N universe symbols (trial run)")
    ap.add_argument("--symbols", help="comma-separated Fyers symbols instead of the universe")
    ap.add_argument("--start", type=date.fromisoformat, help="override history start (YYYY-MM-DD)")
    ap.add_argument("--skip-universe", action="store_true", help="reuse existing universe file")
    args = ap.parse_args(argv)

    setup_logging()
    cfg = get_config()
    dc = cfg.settings.data
    paths = DataPaths(cfg)
    start = args.start or dc.history_start

    try:
        fd = open_data_client(cfg)
    except TokenMissingError as e:
        print(f"[STOP] {e}")
        return 1

    # 1. universe
    if args.skip_universe and paths.universe.exists():
        uni = pipeline.load_universe(paths)
        print(f"[1/6] Universe: {len(uni)} symbols (existing file)")
    else:
        uni = pipeline.refresh_universe(paths, cfg)
        print(
            f"[1/6] Universe: {len(uni)} symbols  (instrument types seen: {uni.attrs.get('type_counts')})"
        )

    # 2. indices (also the calendar source)
    end = latest_completed_session(TradingCalendar.from_config())
    s = pipeline.load_indices(fd, paths, start, end, cfg)
    print(f"[2/6] Indices: fetched {s.fetched}, up-to-date {s.skipped}, failed {len(s.failed)}")
    for sym, err in s.failed.items():
        print(f"      ! {sym}: {err[:120]}")

    # 3. calendar
    n = pipeline.build_calendar(paths, cfg)
    cal = TradingCalendar.from_config()
    end = latest_completed_session(cal)
    print(f"[3/6] Calendar: {n} sessions {cal.first} -> {cal.last}; target last session {end}")

    # 4. equities
    if args.symbols:
        symbols = [x.strip() for x in args.symbols.split(",") if x.strip()]
    else:
        symbols = list(uni["symbol"])
        if args.limit:
            symbols = symbols[: args.limit]
    t0 = time.monotonic()

    def progress(i: int, total: int, sm: RunSummary) -> None:
        el = time.monotonic() - t0
        eta = el / i * (total - i)
        print(
            f"      [{i:>5}/{total}] fetched {sm.fetched}  up-to-date {sm.skipped}  "
            f"no-data {sm.empty}  failed {len(sm.failed)}  | API today {fd.throttler.usage['count']}"
            f"  | elapsed {fmt_dur(el)}  ETA {fmt_dur(eta)}",
            flush=True,
        )

    print(
        f"[4/6] Equities: {len(symbols)} symbols from {start} to {end} (Ctrl+C is safe; re-run resumes)"
    )
    try:
        es = pipeline.load_equities(fd, paths, symbols, start, end, progress, cfg)
    except KeyboardInterrupt:
        print("\n[PAUSED] Stopped by you. Re-run the same command to resume.")
        return 130
    if es.failed:
        print(f"      {len(es.failed)} symbols failed (first 10):")
        for sym, err in list(es.failed.items())[:10]:
            print(f"      ! {sym}: {err[:120]}")
    if es.aborted:
        print(f"[STOP] {es.aborted}")
        return 2

    # 5. validation
    iss = pipeline.validate_all(paths, symbols, end, cfg)
    counts = iss.groupby(["check", "severity"]).size().to_dict() if len(iss) else {}
    print(f"[5/6] Validation: {len(iss)} issues {counts}")

    # 6. database
    tables = build_database(paths)
    print(f"[6/6] DuckDB: {tables}")
    print("\nDone. Now run:  python scripts\\p1_exit_test.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
