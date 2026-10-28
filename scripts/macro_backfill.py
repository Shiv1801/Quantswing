"""D39: download the cross-asset drivers Fyers offers (sector indices, 10-year G-Sec index,
gold ETF) and build the macro spill-over factors.

Needs a Fyers login for today (``python scripts/fyers_login.py``) and the P1/P2 data.
Resumable; daily updates then happen automatically in ``run_eod.py`` (indices step).

Usage::

    python scripts/macro_backfill.py
"""

from __future__ import annotations

import sys
import time

from quantswing.alpha.research import build_family
from quantswing.broker.session import open_data_client
from quantswing.core.calendar import TradingCalendar
from quantswing.core.config import get_config
from quantswing.core.logging import setup_logging
from quantswing.core.types import TokenMissingError
from quantswing.ingest.fyers_history import HistoryLoader, latest_completed_session
from quantswing.ingest.fyers_universe import download_symbol_master
from quantswing.ingest.macro import SECTORS, find_drivers, master_tickers, save_symbols
from quantswing.store.panels import FactorStore
from quantswing.store.repositories import DataPaths, JsonState

MIN_ROWS = 250  # a driver needs about a year of history to estimate sensitivities


def main() -> int:
    """Find → download → keep drivers with enough history → build factors."""
    setup_logging()
    t0 = time.monotonic()
    cfg = get_config()
    paths = DataPaths(cfg)
    try:
        fd = open_data_client(cfg)
    except TokenMissingError as e:
        print(f"[STOP] {e}")
        return 1
    print("[1/3] Reading the Fyers symbol master...", flush=True)
    tickers = master_tickers(download_symbol_master(cfg.settings.data.universe))
    drivers = find_drivers(tickers)
    wanted = [*SECTORS, "GSEC", "GOLD"]
    for name in wanted:
        print(f"      {name:<12} {drivers.get(name, '- not listed by Fyers')}")
    cal = TradingCalendar.from_config()
    end = latest_completed_session(cal)
    start = cfg.settings.data.history_start
    print(f"[2/3] Downloading {len(drivers)} drivers {start} -> {end}...", flush=True)
    loader = HistoryLoader(fd, paths.idx_store(), JsonState(paths.state), kind="idx", cfg=cfg)
    summ = loader.run(list(drivers.values()), start, end)
    for sym, err in summ.failed.items():
        print(f"      ! {sym}: {err[:120]}")
    store = paths.idx_store()
    kept: dict[str, str] = {}
    for name, sym in drivers.items():
        df = store.read(sym)
        n = 0 if df is None else len(df)
        first = "" if df is None or df.empty else str(df["date"].min())[:10]
        ok = n >= MIN_ROWS
        print(
            f"      {name:<12} {sym:<32} {n:>5} days from {first or '-':<10} {'OK' if ok else 'too short - skipped'}"
        )
        if ok:
            kept[name] = sym
    save_symbols(paths.macro_symbols, kept)
    print(f"      saved {len(kept)} drivers -> {paths.macro_symbols}")
    if not FactorStore(paths.factors).path("composite", "COMPOSITE").exists():
        print(
            "[3/3] No factor store yet - run run_factors.py and p2_ic_report.py, then re-run this."
        )
        return 1
    print("[3/3] Building the macro spill-over factors (a few minutes)...", flush=True)
    names = build_family(
        paths, cfg, "macro", progress=lambda n: print(f"      built {n}", flush=True)
    )
    if not names:
        print("[STOP] No macro factors could be built (no driver data).")
        return 1
    print(
        f"\nDONE in {time.monotonic() - t0:.0f}s: {len(kept)} drivers, factors {', '.join(names)}"
    )
    print("Next: python scripts\\run_macro_test.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
