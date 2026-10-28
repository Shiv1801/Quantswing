"""Optional: answer Telegram commands instantly while this window is open (Ctrl+C stops).
Without it, commands are handled at the next daily run.

Usage::

    python scripts/telegram_bot.py
"""

from __future__ import annotations

import sys
import time

from quantswing.alerts.telegram_bot import Telegram, process_commands
from quantswing.core.config import get_config
from quantswing.core.logging import setup_logging
from quantswing.execution.paper_broker import PaperBook
from quantswing.store.repositories import DataPaths


def main() -> int:
    """Long-poll loop."""
    setup_logging()
    cfg = get_config()
    tg = Telegram.from_env(cfg.paper.telegram.timeout_s + 30)
    if not tg.enabled:
        print("[STOP] Telegram not set up - run: python scripts\\telegram_setup.py")
        return 1
    book = PaperBook(DataPaths(cfg).paper_db)
    print("Listening for Telegram commands (Ctrl+C to stop)...")
    try:
        while True:
            n = process_commands(tg, book)
            if n:
                print(f"  handled {n} command(s)")
            time.sleep(2)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
