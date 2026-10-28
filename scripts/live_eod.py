"""D40 end-of-day update for live paper trading (the app's "Run end-of-day update" button).

1. Data: Fyers prices + NSE data (same as run_eod.py) — needs today's Fyers login.
2. Research: factors → the D37 10-day score.
3. Trade engine: replay the new session(s) exactly like the backtest (buys at the open,
   SL / target / TSL during the day, marks at the close), then plan the next session.
4. The action list for the next session → the app's Actions tab + Telegram.

Usage::

    python scripts/live_eod.py              # everything
    python scripts/live_eod.py --no-data    # data already updated today
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
import traceback
from pathlib import Path

import backfill  # scripts/ is on sys.path when run as a script
import nse_backfill

from quantswing.alerts.telegram_bot import Telegram
from quantswing.broker.fyers_auth import TokenStore, get_valid_token
from quantswing.core.config import get_config
from quantswing.core.logging import setup_logging
from quantswing.core.types import TokenMissingError
from quantswing.live.book import LiveBook
from quantswing.live.planner import eod_step, eod_text, research
from quantswing.store.repositories import DataPaths


class Tee:
    """Mirror prints into the log file (the app shows this file while the run is going)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("w", encoding="utf-8")
        self.out = sys.__stdout__

    def write(self, s: str) -> int:
        self.f.write(s)
        self.f.flush()
        if self.out is not None:
            with contextlib.suppress(Exception):
                self.out.write(s)
        return len(s)

    def flush(self) -> None:
        self.f.flush()

    def isatty(self) -> bool:
        return False


def main(argv: list[str] | None = None) -> int:
    """Run the end-of-day update."""
    ap = argparse.ArgumentParser(description="QuantSwing live paper: end-of-day update")
    ap.add_argument("--no-data", action="store_true", help="skip the data download")
    args = ap.parse_args(argv)
    cfg = get_config()
    paths = DataPaths(cfg)
    sys.stdout = Tee(paths.live_eod_log)
    setup_logging()
    book = LiveBook(paths.live_db)
    rid = book.run_start("eod")
    tg = Telegram.from_env(cfg.paper.telegram.timeout_s)
    t0 = time.monotonic()
    try:
        if not args.no_data:
            try:
                get_valid_token(TokenStore.from_config(cfg))
            except TokenMissingError:
                msg = "No Fyers login for today - log in from the app (Login page), then run again."
                print(f"[STOP] {msg}")
                book.run_end(rid, "stopped", msg)
                return 1
            print("[1/3] Updating data (Fyers + NSE)...", flush=True)
            rc = backfill.main([])
            if rc != 0:
                raise RuntimeError(f"Fyers data update failed (code {rc})")
            with contextlib.suppress(Exception):  # NSE hiccups never block the run
                nse_backfill.main(["--no-sectors"])  # sectors: monthly, not needed daily
        else:
            print("[1/3] Data update skipped (--no-data).")
        print("[2/3] Research...", flush=True)
        comp, ic = research(paths, cfg, lambda m: print(f"      {m}", flush=True))
        print("[3/3] Trade engine...", flush=True)
        rep = eod_step(paths, cfg, book, comp, ic, lambda m: print(f"      {m}", flush=True))
        text = eod_text(rep, book)
        print("\n" + text)
        if cfg.paper.telegram.enabled and tg.enabled:
            tg.send(text)
        print(f"\nDONE in {time.monotonic() - t0:.0f}s")
        book.run_end(rid, "ok", text)
        return 0
    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n[FAILED] {type(e).__name__}: {e}\n{tb}")
        book.run_end(rid, "failed", f"{type(e).__name__}: {e}")
        if tg.enabled:
            tg.send(f"QuantSwing EOD failed: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
