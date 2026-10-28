"""NSE secondary data (D31): delivery %, results calendar, corporate actions, sectors,
surveillance lists, holidays and delisted-stock history. Prices of listed stocks stay Fyers.

Needs the P1 Fyers backfill (calendar + universe). No login needed. Resumable: Ctrl+C is
safe, re-running continues. First run ~3 hours (NSE is polled at 1 request/second);
daily runs take a minute or two (``run_eod.py`` calls this automatically).

Usage::

    python scripts/nse_backfill.py --probe      # 2-minute check that NSE answers + formats parse
    python scripts/nse_backfill.py              # everything (first time ~3 h)
    python scripts/nse_backfill.py --no-sectors # skip the per-stock sector lookups
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta

import pandas as pd

from quantswing.core.config import get_config
from quantswing.core.logging import setup_logging
from quantswing.nse import formats as nf
from quantswing.nse.build import build_all
from quantswing.nse.client import NseClient, NseError
from quantswing.nse.loader import FILE_KINDS, NseLoader, month_chunks
from quantswing.store.database import build_database
from quantswing.store.repositories import DataPaths, connect


def say(msg: str) -> None:
    """Progress line."""
    print(msg, flush=True)


def probe(loader: NseLoader, tickers: list[str]) -> bool:
    """Fetch one recent file of each kind + one API chunk each + 3 sector quotes."""
    ok = True
    ss = loader.sessions()
    recent = [d for d in ss if d <= loader.today - timedelta(days=1)][-3:]
    hosts = loader.nc.http.archive_hosts
    say(f"HTTP backend: {loader.client.backend}")
    for d in reversed(recent):
        for kind, urls, magic in (
            ("bhav", nf.bhav_urls(d, hosts, loader.nc.files.udiff_switch), b"PK"),
            ("mto", nf.mto_urls(d, hosts), None),
            ("surv", nf.surveillance_urls(d, hosts), None),
        ):
            dl = loader.client.download(urls, magic)
            if dl.status == "ok":
                if kind == "bhav":
                    df = nf.parse_bhav(dl.content)
                elif kind == "mto":
                    df = nf.parse_mto(dl.content)
                else:
                    df = nf.parse_surveillance(dl.content, d)
                say(f"  [OK] {kind:<5} {d}  {len(df):>5} rows   {dl.url.rsplit('/', 1)[-1]}")
            else:
                say(f"  [--] {kind:<5} {d}  {dl.status} (HTTP {dl.http})")
                ok = ok and dl.status != "fail"
        break
    a, b = month_chunks(loader.today - timedelta(days=40), loader.today)[0]
    for kind in ("ca", "results", "meetings"):
        try:
            df = loader._api_call(kind, a, b)
            say(f"  [OK] {kind:<8} {a}..{b}  {len(df)} rows")
        except NseError as e:
            say(f"  [XX] {kind:<8} {e}")
            ok = False
    for t in tickers[:3]:
        try:
            b_, s_ = t.split(":", 1)[-1].rsplit("-", 1)
            sec = loader._quote_sector(b_, s_)
            say(f"  [OK] sector   {t}: {sec.get('sector')} / {sec.get('industry')}")
        except NseError as e:
            say(f"  [XX] sector   {t}: {e}")
            ok = False
    say(f"HTTP backend used: {loader.client.backend}")
    if loader.client.last_error:
        say(f"Last network error: {loader.client.last_error}")
    return ok


def data_test(paths: DataPaths, loader: NseLoader, n_universe: int) -> dict[str, bool]:
    """NSE DATA TEST checks."""
    ss = loader.sessions()
    c = connect(paths.duckdb, read_only=True)
    try:
        q = lambda sql: c.execute(sql).fetchone()[0]  # noqa: E731
        start = loader.nc.start.isoformat()
        rows = q(f"SELECT count(*) FROM prices_daily WHERE date >= DATE '{start}' AND NOT delisted")
        with_del = q(
            f"SELECT count(*) FROM prices_daily WHERE date >= DATE '{start}' AND NOT delisted "
            "AND delivery_pct IS NOT NULL"
        )
        sectors = q("SELECT count(DISTINCT symbol) FROM sectors")
        events = q("SELECT count(*) FROM results_events")
        delisted = q("SELECT count(*) FROM delisted")
        surv_days = q("SELECT count(*) FROM surveillance_dates")
        divs = q("SELECT count(*) FROM dividends")
        last_year = (loader.today - timedelta(days=365)).isoformat()
        ev_names = q(
            "SELECT count(DISTINCT symbol) FROM results_events "
            f"WHERE event_date >= DATE '{last_year}'"
        )
        by_year = c.execute(
            "SELECT year(event_date) AS y, "
            "count(*) FILTER (WHERE source = 'filing') AS filing, "
            "count(*) FILTER (WHERE source = 'meeting') AS meeting "
            "FROM results_events GROUP BY 1 ORDER BY 1"
        ).fetchall()
    finally:
        c.close()
    n_files = {
        k: len(list((paths.nse / k).glob("*/*.parquet"))) if (paths.nse / k).exists() else 0
        for k in FILE_KINDS
    }
    months = len(month_chunks(loader.nc.start, loader.today))
    api_done = {
        k: (
            len(list((paths.nse / "api" / k).glob("*.parquet")))
            if (paths.nse / "api" / k).exists()
            else 0
        )
        for k in ("ca", "results", "meetings")
    }
    say("\nNSE data summary")
    say(f"  bhavcopy sessions      {n_files['bhav']:>7,} / {len(ss):,}")
    say(f"  delivery (MTO)         {n_files['mto']:>7,} / {len(ss):,}")
    say(f"  surveillance sessions  {surv_days:>7,}  (from {loader.state.get('surv_first', '-')})")
    for k, v in api_done.items():
        say(f"  {k:<22} {v:>7,} / {months} months")
    say(f"  results events         {events:>7,}  ({ev_names:,} stocks reported in the last year)")
    say(
        "    by year (results feed / board-meeting fallback): "
        + ", ".join(f"{y}: {f:,}/{m:,}" for y, f, m in by_year)
    )
    say(f"  dividends              {divs:>7,}")
    say(f"  sectors                {sectors:>7,} / {n_universe:,} stocks")
    say(f"  delisted stocks added  {delisted:>7,}")
    frac = with_del / rows if rows else 0.0
    say(f"  prices with delivery % {frac:>7.1%}")
    return {
        "bhavcopy coverage >= 95%": n_files["bhav"] >= 0.95 * len(ss),
        "delivery coverage >= 90%": n_files["mto"] >= 0.90 * len(ss),
        "corporate actions >= 95% of months": api_done["ca"] >= 0.95 * months,
        "results >= 95% of months": api_done["results"] >= 0.95 * months,
        "board meetings >= 95% of months": api_done["meetings"] >= 0.95 * months,
        "sectors >= 90% of universe": sectors >= 0.90 * n_universe,
        "delivery % on >= 80% of price rows": frac >= 0.80,
        "surveillance file found": surv_days > 0,
        "results for >= 60% of stocks (last year)": ev_names >= 0.60 * n_universe,
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on PASS."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe", action="store_true", help="quick connectivity + format check")
    ap.add_argument("--no-sectors", action="store_true", help="skip per-stock sector lookups")
    ap.add_argument("--no-files", action="store_true", help="skip the daily archive files")
    ap.add_argument("--no-api", action="store_true", help="skip corporate actions / results")
    ap.add_argument(
        "--refresh-ca", action="store_true", help="re-download all corporate actions (~3 min)"
    )
    args = ap.parse_args(argv)
    setup_logging()
    cfg = get_config()
    paths = DataPaths(cfg)
    if not cfg.nse.enabled:
        say("NSE data is disabled (config/nse.yaml: enabled: false).")
        return 0
    if not (paths.calendar.exists() and paths.universe.exists()):
        say("[STOP] Run the Fyers backfill first: python scripts\\backfill.py")
        return 1
    tickers = pd.read_parquet(paths.universe)["symbol"].tolist()
    client = NseClient(cfg.nse.http, ca_dir=paths.nse_state.parent)
    loader = NseLoader(paths, cfg, client, progress=say)
    if args.probe:
        say("NSE probe (one file / chunk of each kind)...")
        ok = probe(loader, tickers)
        say(f"\nNSE PROBE: {'PASS' if ok else 'FAIL'}")
        if not ok:
            say("If everything says XX/fail: NSE is blocking this network - try another")
            say("network (e.g. home Wi-Fi / phone hotspot) and run the probe again.")
        return 0 if ok else 1
    t0 = time.monotonic()
    ss = loader.sessions()
    todo = sum(1 for k in FILE_KINDS for d in ss if not loader.file_path(k, d).exists())
    say(f"NSE backfill: {len(ss):,} sessions since {cfg.nse.start}; up to {todo:,} files to fetch")
    say(
        f"  (~{todo * cfg.nse.http.min_interval_s / 3600:.1f} h at 1 file/s first time; Ctrl+C safe)"
    )
    if args.refresh_ca:
        for f in (paths.nse / "api" / "ca").glob("*.parquet"):
            f.unlink()
        say("Corporate actions will be downloaded again.")
    summ = loader.run(
        files=not args.no_files,
        api=not args.no_api,
        sectors=not args.no_sectors,
        tickers=tickers,
    )
    if summ.aborted:
        say(f"\n[STOP] {summ.aborted}")
        say("Nothing is lost: run the same command again later (or from another network).")
    say("Building NSE tables (symbol map, delisted stocks, events, dividends)...")
    rep = build_all(paths, cfg, progress=say)
    say(
        f"  delisted stocks: {rep.delisted} ({rep.delisted_rows:,} rows, "
        f"{rep.delisted_adjustments} split/bonus adjustments, "
        f"{rep.delisted_mergers} were mergers/schemes -> exit at full value)"
    )
    say("Rebuilding the database...")
    build_database(paths, cfg)
    checks = data_test(paths, loader, len(tickers))
    say("\nChecks:")
    for k, v in checks.items():
        say(f"   [{'OK' if v else 'XX'}] {k}")
    passed = all(checks.values()) and not summ.aborted
    say(f"\n{summ.requests:,} NSE requests in {(time.monotonic() - t0) / 60:.0f} min")
    say(f"\nNSE DATA TEST: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
