"""Daily EOD data update (after 15:30 IST): Fyers prices (refresh universe, fetch the last few
days for every symbol, re-validate) then the NSE data (D31: new delivery/surveillance files,
latest corporate actions and results, monthly sector refresh) and a DuckDB rebuild.

Usage::

    python scripts/run_eod.py            # Fyers + NSE
    python scripts/run_eod.py --no-nse   # Fyers only
"""

from __future__ import annotations

import sys

import backfill  # scripts/ is on sys.path when run as a script
import nse_backfill

if __name__ == "__main__":
    argv = sys.argv[1:]
    skip_nse = "--no-nse" in argv
    argv = [a for a in argv if a != "--no-nse"]
    rc = backfill.main(argv)
    if rc == 0 and not skip_nse:
        print("\n--- NSE data ---", flush=True)
        nse_backfill.main([])  # NSE hiccups never fail the EOD run
    sys.exit(rc)
