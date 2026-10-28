"""Paper account status + the P6 exit test (2 weeks unattended, no crashes).

Usage::

    python scripts/paper_status.py              # status + exit test (includes a backtest replay)
    python scripts/paper_status.py --no-replay  # skip the replay (faster)
"""

from __future__ import annotations

import argparse
import sys

from quantswing.alerts.telegram_bot import status_text
from quantswing.core.config import get_config
from quantswing.core.logging import setup_logging
from quantswing.execution.paper_broker import PaperBook
from quantswing.paper.status import paper_status
from quantswing.store.repositories import DataPaths


def main(argv: list[str] | None = None) -> int:
    """Returns 0 when the P6 exit test passes."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-replay", action="store_true")
    args = ap.parse_args(argv)
    setup_logging()
    cfg = get_config()
    paths = DataPaths(cfg)
    book = PaperBook(paths.paper_db)
    print(status_text(book))
    pos = book.positions()
    if not pos.empty:
        print("\n" + pos[["symbol", "qty", "last_px", "weight", "since"]].to_string(index=False))
    st = paper_status(paths, cfg, book, replay=not args.no_replay)
    print(
        f"\nStarted {st.start} | sessions {st.sessions} | return {st.ret:+.2%} | max DD {st.max_dd:.1%}"
    )
    if st.replay_gap == st.replay_gap:
        print(f"Paper vs backtest replay: max gap {st.replay_gap:.3%}")
    if st.mismatches:
        print(f"Reconciliation mismatches: {st.mismatches}")
    print("\nChecks:")
    for k, v in st.checks.items():
        print(f"   [{'OK' if v else 'XX'}] {k}")
    print(f"\nP6 EXIT TEST: {'PASS' if st.passed else 'NOT YET'}")
    return 0 if st.passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
