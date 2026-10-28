"""NSE backfill + daily update (D31). Resumable; every parsed file is cached as Parquet.

Layout under ``paths.nse`` (``data/processed/nse``)::

    bhav/<YYYY>/<YYYYMMDD>.parquet      OHLCV + ISIN, EQ/BE/BZ (all stocks, incl. later-delisted)
    mto/<YYYY>/<YYYYMMDD>.parquet       delivery quantity / %
    surv/<YYYY>/<YYYYMMDD>.parquet      surveillance flags
    api/ca/<from>_<to>.parquet          corporate actions (splits, bonuses, dividends)
    api/results/<from>_<to>.parquet     results broadcast times
    api/meetings/<from>_<to>.parquet    board meetings (results dates announced in advance)
    meetings_upcoming.parquet           forthcoming board meetings
    symbol_change.parquet, holidays.parquet, sectors.parquet

Session list = the observed Fyers calendar (P1) from ``nse.start``. A 404 is recorded as a
miss (holiday / not published) and retried only while recent; a block (403 / network) is a
failure and is retried on the next run.
"""

from __future__ import annotations

import calendar as _cal
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from quantswing.core.clock import now_ist
from quantswing.core.config import AppConfig
from quantswing.core.logging import get_logger
from quantswing.nse import formats as nf
from quantswing.nse.client import NseClient, NseError
from quantswing.store.repositories import DataPaths, JsonState, write_parquet_frame

log = get_logger(__name__)

FILE_KINDS = ("bhav", "mto", "surv")
API_KINDS = ("ca", "results", "meetings")
ABORT_AFTER_FAILS = 25
QUOTE_ENDPOINT = "NextApi/apiClient/GetQuoteApi"


@dataclass
class NseSummary:
    """What a run did."""

    backend: str = ""
    files: dict[str, dict[str, int]] = field(default_factory=dict)  # kind → status → n
    api: dict[str, dict[str, int]] = field(default_factory=dict)
    sectors_ok: int = 0
    sectors_missing: int = 0
    symbol_changes: int = 0
    holidays: int = 0
    surveillance_first: str | None = None
    requests: int = 0
    aborted: str | None = None

    def bump(self, group: str, kind: str, status: str) -> None:
        """Count one outcome."""
        d = self.files if group == "files" else self.api
        d.setdefault(kind, {}).setdefault(status, 0)
        d[kind][status] += 1


def base_symbol(ticker: str) -> tuple[str, str]:
    """``NSE:BAJAJ-AUTO-EQ`` → (``BAJAJ-AUTO``, ``EQ``)."""
    t = ticker.split(":", 1)[-1]
    if "-" not in t:
        return t, ""
    b, s = t.rsplit("-", 1)
    return b, s


def month_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Calendar-month windows covering [start, end]."""
    out: list[tuple[date, date]] = []
    d = date(start.year, start.month, 1)
    while d <= end:
        last = date(d.year, d.month, _cal.monthrange(d.year, d.month)[1])
        out.append((max(d, start), min(last, end)))
        d = last + timedelta(days=1)
    return out


class NseLoader:
    """Downloads, parses and caches NSE data. ``client`` is injectable for tests."""

    def __init__(
        self,
        paths: DataPaths,
        cfg: AppConfig,
        client: NseClient | None = None,
        progress: Callable[[str], None] | None = None,
        today: date | None = None,
    ) -> None:
        self.paths, self.cfg, self.nc = paths, cfg, cfg.nse
        self.client = client or NseClient(self.nc.http, ca_dir=paths.nse_state.parent)
        self.state = JsonState(paths.nse_state)
        self.progress = progress or (lambda _m: None)
        self.today = today or now_ist().date()
        self.summary = NseSummary(backend=self.client.backend)

    # ------------------------------------------------------------------ paths
    def file_path(self, kind: str, d: date) -> Path:
        """Parsed Parquet for a daily file."""
        return self.paths.nse / kind / str(d.year) / f"{d:%Y%m%d}.parquet"

    def api_path(self, kind: str, a: date, b: date) -> Path:
        """Parsed Parquet for an API chunk."""
        return self.paths.nse / "api" / kind / f"{a:%Y%m%d}_{b:%Y%m%d}.parquet"

    def sessions(self) -> list[date]:
        """Observed sessions (P1 calendar) from ``nse.start`` to today."""
        if not self.paths.calendar.exists():
            raise RuntimeError("no calendar yet - run the Fyers backfill (P1) first")
        cal = pd.read_parquet(self.paths.calendar)["date"]
        ds = sorted({pd.Timestamp(x).date() for x in cal})
        return [d for d in ds if self.nc.start <= d <= self.today]

    # ------------------------------------------------------------------ daily files
    def _misses(self, kind: str) -> set[str]:
        return set(self.state.get(f"miss_{kind}", []))

    def _fetch_file(self, kind: str, d: date) -> str:
        hosts = self.nc.http.archive_hosts
        if kind == "bhav":
            dl = self.client.download(nf.bhav_urls(d, hosts, self.nc.files.udiff_switch), b"PK")
        elif kind == "mto":
            dl = self.client.download(nf.mto_urls(d, hosts))
        else:
            dl = self.client.download(nf.surveillance_urls(d, hosts))
        if dl.status != "ok":
            return dl.status
        try:
            if kind == "bhav":
                df = nf.parse_bhav(dl.content)
            elif kind == "mto":
                df = nf.parse_mto(dl.content)
            else:
                df = nf.parse_surveillance(dl.content, d)
        except Exception as e:
            log.warning("nse parse failed", extra={"kind": kind, "date": str(d), "error": repr(e)})
            return "bad"
        if df.empty:
            return "bad"
        if kind in ("bhav", "mto"):
            file_dates = set(df["date"].dropna())
            if file_dates and file_dates != {d}:
                log.warning("nse file date mismatch", extra={"kind": kind, "date": str(d)})
                return "bad"
            df["date"] = d
        write_parquet_frame(df, self.file_path(kind, d))
        if self.nc.storage.keep_raw:
            raw = self.paths.nse_raw / kind / str(d.year) / dl.url.rsplit("/", 1)[-1]
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(dl.content)
        return "ok"

    def _surveillance_first(self, sessions: Sequence[date]) -> date | None:
        """First month with a surveillance file (probe the first session of each month)."""
        known = self.state.get("surv_first")
        if known:
            return date.fromisoformat(known)
        firsts: dict[tuple[int, int], date] = {}
        for d in sessions:
            if d >= self.nc.files.surveillance_probe_from:
                firsts.setdefault((d.year, d.month), d)
        for d in sorted(firsts.values()):
            st = self._fetch_file("surv", d)
            if st == "ok":
                first = date(d.year, d.month, 1)
                self.state.set("surv_first", first.isoformat())
                self.progress(f"  surveillance files start {first:%b %Y}")
                return first
            if st == "fail":
                return None  # blocked: probe again next run
        return None

    def run_files(
        self, kinds: Sequence[str] = FILE_KINDS, sessions: Sequence[date] | None = None
    ) -> None:
        """Download + parse every missing daily file."""
        ss = list(sessions) if sessions is not None else self.sessions()
        enabled = {
            "bhav": self.nc.files.bhav,
            "mto": self.nc.files.delivery,
            "surv": self.nc.files.surveillance,
        }
        recent = self.today - timedelta(days=self.nc.files.retry_recent_misses_days)
        for kind in kinds:
            if not enabled.get(kind, False):
                continue
            todo = ss
            if kind == "surv":
                first = self._surveillance_first(ss)
                if first is None:
                    self.summary.bump("files", kind, "unavailable")
                    continue
                self.summary.surveillance_first = first.isoformat()
                todo = [d for d in ss if d >= first]
            misses = self._misses(kind)
            pending = [
                d
                for d in todo
                if not self.file_path(kind, d).exists()
                and not (d.isoformat() in misses and d < recent)
            ]
            self.progress(f"  {kind}: {len(todo) - len(pending)} cached, {len(pending)} to fetch")
            fails = 0
            for n, d in enumerate(pending, 1):
                st = self._fetch_file(kind, d)
                self.summary.bump("files", kind, st)
                if st in ("miss", "bad"):
                    misses.add(d.isoformat())
                fails = fails + 1 if st == "fail" else 0
                if fails >= ABORT_AFTER_FAILS:
                    self.summary.aborted = f"{kind}: {fails} consecutive failures (NSE blocking?)"
                    break
                if n % 100 == 0:
                    self.state.set(f"miss_{kind}", sorted(misses))
                    self.progress(f"  {kind}: {n}/{len(pending)} ({d})")
            self.state.set(f"miss_{kind}", sorted(misses))
            if self.summary.aborted:
                return

    # ------------------------------------------------------------------ API chunks
    def _api_call(self, kind: str, a: date, b: date) -> pd.DataFrame:
        fmt = "%d-%m-%Y"
        rng = {"from_date": a.strftime(fmt), "to_date": b.strftime(fmt)}
        if kind == "ca":
            return nf.parse_corp_actions(
                self.client.api("corporates-corporateActions", {"index": "equities", **rng})
            )
        if kind == "results":
            return nf.parse_results(
                self.client.api(
                    "corporates-financial-results",
                    {"index": "equities", "period": "Quarterly", **rng},
                )
            )
        return nf.parse_board_meetings(
            self.client.api("corporate-board-meetings", {"index": "equities", **rng})
        )

    def run_api(self, kinds: Sequence[str] = API_KINDS) -> None:
        """Fetch every missing monthly chunk (the most recent ones are always refreshed)."""
        chunks = month_chunks(self.nc.start, self.today)
        refresh = {c for c in chunks[-self.nc.api.refresh_recent_chunks :]} if chunks else set()
        for kind in kinds:
            done = 0
            fails = 0
            for a, b in chunks:
                p = self.api_path(kind, a, b)
                if p.exists() and (a, b) not in refresh:
                    done += 1
                    continue
                # a chunk that ended before today is complete; the current one is re-fetched
                for old in p.parent.glob(f"{a:%Y%m%d}_*.parquet") if p.parent.exists() else []:
                    if old != p:
                        old.unlink()
                try:
                    df = self._api_call(kind, a, b)
                except NseError as e:
                    self.summary.bump("api", kind, "fail")
                    log.warning("nse api chunk failed", extra={"kind": kind, "error": str(e)})
                    fails += 1
                    if fails >= ABORT_AFTER_FAILS // 5 and done == 0:
                        self.summary.aborted = f"{kind}: NSE API refused {fails} requests"
                        return
                    continue
                write_parquet_frame(df, p)
                self.summary.bump("api", kind, "ok")
                done += 1
            self.progress(f"  {kind}: {done}/{len(chunks)} months")
        try:
            up = nf.parse_board_meetings(
                self.client.api("corporate-board-meetings", {"index": "equities"})
            )
            write_parquet_frame(up, self.paths.nse / "meetings_upcoming.parquet")
        except NseError as e:
            log.warning("upcoming meetings failed", extra={"error": str(e)})

    # ------------------------------------------------------------------ small lists
    def run_symbol_change(self) -> None:
        """Symbol-change list (needed to map old NSE symbols onto current tickers)."""
        dl = self.client.download(nf.symbol_change_urls(self.nc.http.archive_hosts))
        if dl.status == "ok":
            df = nf.parse_symbol_change(dl.content)
            write_parquet_frame(df, self.paths.nse / "symbol_change.parquet")
            self.summary.symbol_changes = len(df)

    def run_holidays(self) -> None:
        """Trading holidays (current + next year as published)."""
        try:
            df = nf.parse_holidays(self.client.api("holiday-master", {"type": "trading"}))
        except NseError as e:
            log.warning("holidays failed", extra={"error": str(e)})
            return
        f = self.paths.nse / "holidays.parquet"
        if f.exists():
            df = pd.concat([pd.read_parquet(f), df]).drop_duplicates("date")
        write_parquet_frame(df.sort_values("date"), f)
        self.summary.holidays = len(df)

    # ------------------------------------------------------------------ sectors
    def _quote_sector(self, sym: str, series: str) -> dict[str, Any]:
        params = {
            "functionName": "getSymbolData",
            "marketType": "N",
            "series": series or "EQ",
            "symbol": sym,
        }
        sec = nf.parse_quote_sector(self.client.api(QUOTE_ENDPOINT, params))
        if not sec.get("sector"):
            sec = nf.parse_quote_sector(self.client.api("quote-equity", {"symbol": sym}))
        return sec

    def run_sectors(self, tickers: Sequence[str]) -> None:
        """Sector / industry per stock (refreshed every ``sector_refresh_days``)."""
        f = self.paths.nse / "sectors.parquet"
        old = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["symbol", "as_of"])
        fresh_after = self.today - timedelta(days=self.nc.api.sector_refresh_days)
        have = {
            s: pd.Timestamp(a).date()
            for s, a in zip(old["symbol"], old["as_of"], strict=False)
            if pd.notna(a)
        }
        todo = [t for t in tickers if have.get(t, date.min) < fresh_after]
        rows: dict[str, dict[str, Any]] = {
            str(r["symbol"]): {str(k): v for k, v in r.items()} for r in old.to_dict("records")
        }
        self.progress(f"  sectors: {len(tickers) - len(todo)} fresh, {len(todo)} to fetch")
        fails = 0
        for n, t in enumerate(todo, 1):
            b, s = base_symbol(t)
            try:
                sec = self._quote_sector(b, s)
                fails = 0
            except NseError:
                sec = {}
                fails += 1
                if fails >= ABORT_AFTER_FAILS:
                    self.summary.aborted = "sectors: NSE quote API keeps refusing"
                    break
            if sec.get("sector"):
                rows[t] = {"symbol": t, **sec, "as_of": self.today}
            if n % 50 == 0:
                write_parquet_frame(pd.DataFrame(list(rows.values())), f)
                self.progress(f"  sectors: {n}/{len(todo)}")
        if rows:
            write_parquet_frame(pd.DataFrame(list(rows.values())), f)
        self.summary.sectors_ok = sum(1 for t in tickers if t in rows)
        self.summary.sectors_missing = len(tickers) - self.summary.sectors_ok

    # ------------------------------------------------------------------ all
    def run(
        self,
        *,
        files: bool = True,
        api: bool = True,
        sectors: bool = True,
        tickers: Sequence[str] = (),
        sessions: Sequence[date] | None = None,
    ) -> NseSummary:
        """Everything that is missing, in dependency order."""
        self.progress("NSE: symbol changes + holidays")
        self.run_symbol_change()
        self.run_holidays()
        if files:
            self.progress("NSE: daily files (bhavcopy, delivery, surveillance)")
            self.run_files(sessions=sessions)
        if api and not self.summary.aborted:
            self.progress("NSE: corporate actions, results, board meetings")
            self.run_api()
        if sectors and tickers and not self.summary.aborted:
            self.progress("NSE: sectors")
            self.run_sectors(tickers)
        self.summary.requests = self.client.requests
        return self.summary
