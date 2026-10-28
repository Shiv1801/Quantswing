"""D40 live paper update: one poll of live prices for holdings and today's planned buys.

Windows runs this every few minutes during market hours (``install_live_task.py``); the
app's "Update live prices now" button runs it once. Outside market hours it exits at once.

Usage::

    python scripts/live_poll.py           # one update
    python scripts/live_poll.py --loop    # keep updating every poll_minutes until 15:35
    python scripts/live_poll.py --force   # update even outside market hours (testing)
"""

from __future__ import annotations

import argparse
import sys
import time as systime
import traceback
from datetime import time

from quantswing.alerts.telegram_bot import Telegram
from quantswing.broker.session import open_data_client
from quantswing.core.calendar import TradingCalendar
from quantswing.core.clock import now_ist
from quantswing.core.config import get_config
from quantswing.core.logging import setup_logging
from quantswing.core.types import TokenMissingError
from quantswing.live.book import LiveBook
from quantswing.live.poller import OPEN, poll_once
from quantswing.store.repositories import DataPaths


def log(paths: DataPaths, msg: str) -> None:
    """Append one line to the poll log (and print it)."""
    line = f"{now_ist():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    paths.live_poll_log.parent.mkdir(parents=True, exist_ok=True)
    with paths.live_poll_log.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def once(force: bool = False) -> int:
    """One poll. Returns 0 (also when the market is closed)."""
    cfg = get_config()
    paths = DataPaths(cfg)
    book = LiveBook(paths.live_db)
    tg = Telegram.from_env(cfg.paper.telegram.timeout_s)
    now = now_ist()
    cal = TradingCalendar.from_config()
    if not force and (not cal.is_trading_day(now.date()) or not OPEN <= now.time() <= time(15, 40)):
        return 0  # outside market hours: nothing to do (no log spam)
    try:
        fd = open_data_client(cfg)
    except Exception as err:  # no token today, or .env incomplete
        if not isinstance(err, TokenMissingError):
            log(paths, f"cannot open the Fyers client: {type(err).__name__}")
            return 1
        day = str(now_ist().date())
        if book.event(day, "-", "login", "No Fyers login today", f"{day}|login") and tg.enabled:
            tg.send("QuantSwing: no Fyers login today - open the app and click Login.")
        log(paths, "no Fyers login for today - skipped")
        return 0
    rid = book.run_start("poll")
    try:
        rep = poll_once(book, cfg, fd.quotes, tg=tg, force=force)
        log(paths, f"{rep.status}; {rep.symbols} symbols; {len(rep.events)} new events")
        for e in rep.events:
            log(paths, f"  {e}")
        book.run_end(rid, "ok", rep.status)
        return 0
    except Exception as e:
        log(paths, f"FAILED {type(e).__name__}: {e}")
        book.run_end(rid, "failed", traceback.format_exc()[-1500:])
        return 1


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description="QuantSwing live paper update")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    setup_logging()
    if not args.loop:
        return once(args.force)
    cfg = get_config().paper.live
    while now_ist().time() <= cfg.poll_end:
        once(args.force)
        systime.sleep(cfg.poll_minutes * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
