"""P6 daily paper run (evening, after NSE publishes its daily files; see install_schedule.py).

1. Telegram commands (/pause, /flatten yes, ...) are applied.
2. Fyers login: if today's token is missing, the login link is sent on Telegram and the
   run waits for you to paste the redirect address back (or run fyers_login.py yourself).
3. Data: Fyers prices + NSE data (same as run_eod.py).
4. Research: factors → composite → regime.
5. Paper engine: fills at today's open, dividends, mark at the close, plan for the next
   open (catch-up is automatic if the PC was off). Report on Telegram + logs.

Usage::

    python scripts/paper_daily.py              # the full daily run
    python scripts/paper_daily.py --no-update  # skip the data download (data already fresh)
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import traceback
from pathlib import Path

import backfill  # scripts/ is on sys.path when run as a script
import nse_backfill

from quantswing.alerts.telegram_bot import Telegram, process_commands, wait_for_auth
from quantswing.broker.fyers_auth import (
    TokenStore,
    build_login_url,
    get_valid_token,
    login_with_input,
)
from quantswing.core.config import get_config, get_secrets
from quantswing.core.logging import setup_logging
from quantswing.core.types import TokenMissingError
from quantswing.execution.paper_broker import PaperBook
from quantswing.monitor.reports import eod_text
from quantswing.paper.engine import research_refresh, step
from quantswing.store.repositories import DataPaths


class Tee:
    """Mirror prints into the log file when run unattended."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("a", encoding="utf-8")
        self.out = sys.__stdout__

    def write(self, s: str) -> int:
        self.f.write(s)
        self.f.flush()
        if self.out is not None:
            with contextlib.suppress(Exception):  # no console under Task Scheduler
                self.out.write(s)
        return len(s)

    def flush(self) -> None:
        self.f.flush()

    def isatty(self) -> bool:
        return False


def ensure_token(cfg, book: PaperBook, tg: Telegram) -> bool:
    """True if a valid Fyers token exists (possibly after a Telegram login)."""
    store = TokenStore.from_config()
    try:
        get_valid_token(store)
        return True
    except TokenMissingError:
        pass
    if not (cfg.paper.login.via_telegram and tg.enabled):
        print("[STOP] No Fyers login for today. Run: python scripts\\fyers_login.py")
        return False
    s = get_secrets()
    url = build_login_url(s.FYERS_CLIENT_ID.get_secret_value(), s.FYERS_REDIRECT_URI)
    print("Waiting for the Fyers login via Telegram...")
    return wait_for_auth(
        tg,
        book,
        url,
        cfg.paper.login.wait_minutes,
        cfg.paper.login.poll_seconds,
        login=lambda text: login_with_input(text, store),
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point: 0 ok, 1 failed, 2 skipped (no login)."""
    ap = argparse.ArgumentParser(description="QuantSwing daily paper run")
    ap.add_argument("--no-update", action="store_true", help="skip the Fyers/NSE download")
    ap.add_argument("--no-research", action="store_true", help="reuse today's factors/regime")
    args = ap.parse_args(argv)
    cfg = get_config()
    if not isinstance(sys.stdout, Tee) and (sys.stdout is None or not sys.stdout.isatty()):
        sys.stdout = Tee(cfg.settings.paths.resolve(cfg.paper.log_file))  # type: ignore[assignment]
    setup_logging()
    paths = DataPaths(cfg)
    book = PaperBook(paths.paper_db)
    tg = (
        Telegram.from_env(cfg.paper.telegram.timeout_s)
        if cfg.paper.telegram.enabled
        else Telegram(None, None)
    )
    rid = book.run_start()
    print(f"\n===== paper run #{rid} =====", flush=True)
    try:
        n = process_commands(tg, book)
        if n:
            print(f"Telegram commands handled: {n}")
        if not args.no_update:
            if not ensure_token(cfg, book, tg):
                book.incident("login", "no Fyers login - run skipped")
                book.run_end(rid, "skipped", 0, "no login")
                return 2
            print("Updating Fyers data...", flush=True)
            if backfill.main([]) != 0:
                raise RuntimeError("Fyers update failed (see output above)")
            print("\nUpdating NSE data...", flush=True)
            if nse_backfill.main([]) != 0:
                print("  (NSE update incomplete - continuing with what is cached)")
        if not args.no_research:
            print("Rebuilding factors, composite and regime...", flush=True)
            research_refresh(paths, cfg, progress=print)
        rep = step(paths, cfg, book)
        text = eod_text(rep, book)
        print("\n" + text)
        book.put("last_report", text)
        tg.send(text)
        book.run_end(rid, "ok", len(rep.sessions))
        book.resolve_incidents()
        return 0
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        book.incident("run_failed", f"{type(e).__name__}: {e}")
        book.run_end(rid, "failed", 0, f"{type(e).__name__}: {e}")
        tg.send(f"Paper run FAILED: {type(e).__name__}: {e}\nDetails: {cfg.paper.log_file}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
