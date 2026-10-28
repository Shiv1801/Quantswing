"""P0 exit test: token obtained + sample history pulled (07 §1).

Steps: ensure today's token (prompts login if missing) -> profile check -> 1y daily candles
for ``settings.smoke_test.symbols`` -> Parquet under ``data/raw/fyers/daily/`` -> print
row counts, first/last dates, API usage -> ``P0 EXIT TEST: PASS`` / ``FAIL``.

Usage::

    python scripts/p0_smoke_test.py
"""

from __future__ import annotations

import sys
from datetime import timedelta

from quantswing.broker.fyers_auth import TokenStore, get_valid_token, make_client
from quantswing.broker.fyers_data import FyersData
from quantswing.broker.throttler import Throttler
from quantswing.core.calendar import TradingCalendar
from quantswing.core.clock import now_ist, today_ist
from quantswing.core.config import get_config
from quantswing.core.logging import get_logger, setup_logging
from quantswing.core.types import BrokerError, TokenMissingError

log = get_logger("scripts.p0_smoke_test")


def safe_name(symbol: str) -> str:
    """``NSE:RELIANCE-EQ`` -> ``NSE_RELIANCE-EQ`` (Windows-safe file name)."""
    return symbol.replace(":", "_").replace("/", "_")


def main() -> int:
    """Run the smoke test. Returns 0 on PASS."""
    setup_logging()
    cfg = get_config()
    st = cfg.settings.smoke_test
    store = TokenStore.from_config(cfg)

    print(f"QuantSwing P0 smoke test - {now_ist():%Y-%m-%d %H:%M:%S %Z}")
    try:
        tok = get_valid_token(store)
    except TokenMissingError as e:
        print(f"[..] {e}\n     Starting login...")
        import fyers_login  # scripts/ is on sys.path when run as a script

        if fyers_login.main([]) != 0:
            print("P0 EXIT TEST: FAIL (login)")
            return 1
        tok = get_valid_token(store)
    print(f"[OK] Token for {tok.client_id[:4]}**** obtained {tok.obtained_at:%Y-%m-%d %H:%M} IST")

    fd = FyersData(make_client(tok.client_id, tok.access_token), Throttler.from_config(cfg), cfg)
    try:
        prof = fd.profile()
        print(f"[OK] Profile: {(prof.get('data') or {}).get('name', '?')}")
    except BrokerError as e:
        print(f"[FAIL] Profile check: {e} - token may be invalid; run fyers_login.py --force")
        print("P0 EXIT TEST: FAIL (profile)")
        return 1

    end = today_ist()
    start = end - timedelta(days=st.lookback_days)
    cal = TradingCalendar.from_config()
    expected = sum(1 for _ in cal.trading_days(start, end))
    out_dir = cfg.settings.paths.resolve(cfg.settings.paths.raw_dir) / st.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = True
    print(f"\n{'symbol':<22}{'rows':>6}{'first':>13}{'last':>13}  file")
    for sym in st.symbols:
        try:
            df = fd.history(sym, start, end, st.resolution)
        except BrokerError as e:
            print(f"{sym:<22}  ERROR {e}")
            ok = False
            continue
        if df.empty:
            print(f"{sym:<22}{0:>6}  (no data)")
            ok = False
            continue
        path = out_dir / f"{safe_name(sym)}.parquet"
        df.to_parquet(path, index=False)
        first, last = df["ts"].iloc[0].date(), df["ts"].iloc[-1].date()
        print(f"{sym:<22}{len(df):>6}{first!s:>13}{last!s:>13}  {path}")

    print(
        f"\nCalendar-stub trading days in window (weekends only; holidays not loaded): {expected}"
    )
    print("Expected rows ~ that minus ~15 NSE holidays, i.e. ~245-250 per symbol.")
    print(f"API usage today: {fd.throttler.usage}")
    print(f"\nP0 EXIT TEST: {'PASS' if ok else 'FAIL'}")
    log.info("p0 smoke test finished", extra={"pass": ok})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
