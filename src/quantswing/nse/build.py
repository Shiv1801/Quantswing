"""Turn cached NSE files into research tables (D31).

1. **Symbol map** — every NSE symbol segment → a ticker. Old symbols follow the symbol-change
   list onto today's Fyers ticker. Symbols that stopped trading ≥ ``delist_confirm_sessions``
   before the latest bhavcopy and are not in the Fyers universe are **delisted**; they get
   their own ticker (a symbol later reused by another company gets a ``~YYYY`` suffix).
2. **Delisted candles** — rebuilt from bhavcopies into ``processed/nse/delisted_eq`` in the
   Fyers candle schema, validated like Fyers data, and back-adjusted with NSE corporate
   actions (+ the D23 gap detector for anything the feed missed).
3. **Events** — results broadcast → event session (after ``results_cutoff`` → next
   session); results board meetings → meeting session + the session it became known.
4. **Dividends** — per ex-date: ₹ amount ÷ the bhavcopy previous close (yield).

Everything is written as Parquet under ``processed/nse/derived``; DuckDB loads it
(:mod:`quantswing.nse.db`).
"""

from __future__ import annotations

import itertools
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from quantswing.core.calendar import TradingCalendar
from quantswing.core.clock import now_ist
from quantswing.core.config import AppConfig
from quantswing.core.logging import get_logger
from quantswing.nse.loader import base_symbol
from quantswing.store.repositories import DataPaths, write_parquet_frame
from quantswing.store.schema import CANDLE_SCHEMA
from quantswing.store.validation import (
    adjustments_frame,
    detect_adjustments,
    issues_frame,
    validate_symbol,
)

log = get_logger(__name__)
SERIES_RANK = {"EQ": 0, "BE": 1, "BZ": 2}
CA_MATCH_DAYS = 3


@dataclass
class BuildReport:
    """Counts for the NSE data test."""

    nse_symbols: int = 0
    mapped_listed: int = 0
    delisted: int = 0
    unmapped_trading: int = 0
    delisted_rows: int = 0
    delisted_adjustments: int = 0
    delisted_mergers: int = 0
    events: int = 0
    events_from_meetings: int = 0
    meetings: int = 0
    dividends: int = 0
    bhav_sessions: int = 0
    mto_sessions: int = 0
    surv_sessions: int = 0


def derived(paths: DataPaths) -> Path:
    """Folder of derived tables."""
    return paths.nse / "derived"


def glob_of(paths: DataPaths, kind: str) -> str | None:
    """DuckDB glob for a daily-file kind (``None`` if nothing cached)."""
    d = paths.nse / kind
    if not d.exists() or not any(d.glob("*/*.parquet")):
        return None
    return str(d / "*" / "*.parquet").replace("\\", "/")


def read_api(paths: DataPaths, kind: str) -> pd.DataFrame:
    """All cached chunks of an API kind."""
    d = paths.nse / "api" / kind
    files = sorted(d.glob("*.parquet")) if d.exists() else []
    frames = [pd.read_parquet(f) for f in files]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- symbol map
def resolve(sym: str, as_of: date, changes: dict[str, list[tuple[date, str]]]) -> str:
    """Follow the symbol-change chain for rows dated before each change."""
    seen = {sym}
    cur, when = sym, as_of
    while True:
        nxt = [(d, n) for d, n in changes.get(cur, []) if d > when]
        if not nxt:
            return cur
        d, n = min(nxt)
        if n in seen:
            return cur
        seen.add(n)
        cur, when = n, d - timedelta(days=1)


def symbol_map(
    stats: pd.DataFrame,
    universe: list[str],
    fy_first: dict[str, date],
    changes_df: pd.DataFrame,
    latest: date,
    sessions: list[date],
    cfg: AppConfig,
) -> pd.DataFrame:
    """Rows: ``nse_symbol, d_from, d_to, ticker, delisted``.

    ``stats``: one row per NSE symbol with ``first``, ``last``, ``n``, ``series`` (last seen).
    """
    sv = cfg.nse.survivorship
    base_to_ticker = {base_symbol(t)[0]: t for t in universe}
    changes: dict[str, list[tuple[date, str]]] = {}
    for o, n, d in zip(
        changes_df.get("old", []),
        changes_df.get("new", []),
        changes_df.get("date", []),
        strict=False,
    ):
        changes.setdefault(str(o), []).append((pd.Timestamp(d).date(), str(n)))
    sess = pd.DatetimeIndex(pd.to_datetime(sessions))
    li = int(sess.searchsorted(pd.Timestamp(latest), side="right")) - 1
    cutoff = sess[max(li - sv.delist_confirm_sessions, 0)].date() if len(sess) else latest
    rows: list[dict[str, Any]] = []
    for r in stats.to_dict("records"):
        sym = str(r["symbol"])
        first, last = pd.Timestamp(r["first"]).date(), pd.Timestamp(r["last"]).date()
        cuts = sorted(d for d, _ in changes.get(sym, []) if first < d <= last)
        bounds = [first, *cuts, last + timedelta(days=1)]
        for a, b in itertools.pairwise(bounds):
            seg_to = b - timedelta(days=1)
            res = resolve(sym, seg_to, changes)
            t = base_to_ticker.get(res)
            if t is not None:
                ff = fy_first.get(t)
                if ff is None or seg_to >= ff - timedelta(days=30):
                    rows.append(
                        dict(nse_symbol=sym, d_from=a, d_to=seg_to, ticker=t, delisted=False)
                    )
                    continue
                res = f"{res}~{seg_to:%Y}"  # symbol reused later by another company
            if seg_to <= cutoff and int(r["n"]) >= sv.min_sessions:
                ser = str(r["series"]) if str(r["series"]) in ("EQ", "BE") else "EQ"
                rows.append(
                    dict(
                        nse_symbol=sym,
                        d_from=a,
                        d_to=seg_to,
                        ticker=f"NSE:{res}-{ser}",
                        delisted=True,
                    )
                )
    cols = ["nse_symbol", "d_from", "d_to", "ticker", "delisted"]
    return pd.DataFrame(rows, columns=cols)


def map_frame(df: pd.DataFrame, smap: pd.DataFrame, sym_col: str, date_col: str) -> pd.Series:
    """Ticker for each row of ``df`` (NaN where unmapped), by symbol and date segment."""
    out = pd.Series(np.nan, index=df.index, dtype="object")
    if df.empty or smap.empty:
        return out
    left = pd.DataFrame(
        {
            "_i": np.arange(len(df)),
            "nse_symbol": df[sym_col].astype(str).to_numpy(),
            "_d": pd.to_datetime(df[date_col]).dt.normalize().to_numpy(),
        }
    )
    right = smap[["nse_symbol", "d_from", "d_to", "ticker"]].assign(
        d_from=lambda x: pd.to_datetime(x["d_from"]), d_to=lambda x: pd.to_datetime(x["d_to"])
    )
    m = left.merge(right, on="nse_symbol", how="inner")
    m = m[(m["_d"] >= m["d_from"]) & (m["_d"] <= m["d_to"])].drop_duplicates("_i")
    vals = out.to_numpy(dtype=object, copy=True)
    vals[m["_i"].to_numpy()] = m["ticker"].to_numpy()
    return pd.Series(vals, index=df.index, dtype="object")


# --------------------------------------------------------------------------- delisted candles
def delisted_candles(
    con: duckdb.DuckDBPyConnection, bhav_glob: str, smap: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Per delisted ticker: candle frame in ``CANDLE_SCHEMA``."""
    dl = smap[smap["delisted"]]
    if dl.empty:
        return {}
    con.register("dmap", dl[["nse_symbol", "d_from", "d_to", "ticker"]])
    q = f"""
        SELECT m.ticker, b.date, b.series, b.open, b.high, b.low, b.close, b.volume
        FROM read_parquet('{bhav_glob}', union_by_name=true) b
        JOIN dmap m ON b.symbol = m.nse_symbol
         AND CAST(b.date AS DATE) BETWEEN CAST(m.d_from AS DATE) AND CAST(m.d_to AS DATE)
    """
    df = con.execute(q).fetchdf()
    con.unregister("dmap")
    if df.empty:
        return {}
    df["rank"] = df["series"].map(SERIES_RANK).fillna(9)
    df = df.sort_values(["ticker", "date", "rank"]).drop_duplicates(["ticker", "date"])
    fetched = pd.Timestamp(now_ist())
    out: dict[str, pd.DataFrame] = {}
    for t, g in df.groupby("ticker"):
        # zero-volume rows = no trade that day (bhavcopy repeats the last price): not a session
        g = g[(g["close"] > 0) & g["open"].notna() & (g["volume"] > 0)]
        if g.empty:
            continue
        d = pd.DatetimeIndex(pd.to_datetime(g["date"]))
        c = pd.DataFrame(
            {
                "date": d.date,
                "ts": d.tz_localize("Asia/Kolkata"),
                "symbol": t,
                "open": g["open"].astype("float64").to_numpy(),
                "high": g["high"].astype("float64").to_numpy(),
                "low": g["low"].astype("float64").to_numpy(),
                "close": g["close"].astype("float64").to_numpy(),
                "volume": g["volume"].fillna(0).astype("int64").to_numpy(),
                "fetched_at": fetched,
            }
        )
        out[str(t)] = c[list(CANDLE_SCHEMA.names)].reset_index(drop=True)
    return out


def ca_adjustments(ca: pd.DataFrame, candles: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Split/bonus/consolidation events for delisted tickers (ex-date snapped to a session)."""
    out: list[dict[str, Any]] = []
    if ca.empty or "ticker" not in ca:
        return out
    ev = ca[ca["kind"].isin(["split", "bonus", "consolidation"]) & ca["ticker"].isin(list(candles))]
    for t, g in ev.groupby("ticker"):
        dates = pd.Index(candles[str(t)]["date"])
        for e in g.drop_duplicates(["ex_date", "kind", "ratio"]).to_dict("records"):
            pos = dates.searchsorted(e["ex_date"])
            if 0 < pos < len(dates):
                out.append(
                    {
                        "symbol": t,
                        "ex_date": dates[pos],
                        "ratio": float(e["ratio"]),
                        "observed": np.nan,
                    }
                )
    return out


REORG_WINDOW = (-365, 180)  # reorg action this many days around the last trade = merger


def delisting_reasons(smap: pd.DataFrame, ca: pd.DataFrame, haircut: float) -> pd.DataFrame:
    """Per delisted ticker: ``merger/scheme`` (exit at the last price — holders are paid in
    cash or acquirer shares) or ``other`` (exit at last price × (1 − haircut))."""
    cols = ["symbol", "reason", "haircut"]
    dl = smap[smap["delisted"]]
    if dl.empty:
        return pd.DataFrame(columns=cols)
    last: dict[str, Any] = {str(k): v for k, v in dl.groupby("ticker")["d_to"].max().items()}
    syms: dict[str, set[str]] = {
        str(t): {str(x) for x in g} for t, g in dl.groupby("ticker")["nse_symbol"]
    }
    reorg: dict[str, list[date]] = {}
    if not ca.empty and "kind" in ca:
        r = ca[ca["kind"] == "reorg"]
        for sym, ex in zip(r["symbol"], r["ex_date"], strict=False):
            reorg.setdefault(str(sym), []).append(pd.Timestamp(ex).date())
    rows = []
    lo, hi = (timedelta(days=d) for d in REORG_WINDOW)
    for t, ld in last.items():
        d_last = pd.Timestamp(ld).date()
        hit = any(d_last + lo <= ex <= d_last + hi for s in syms[str(t)] for ex in reorg.get(s, []))
        rows.append(
            {
                "symbol": t,
                "reason": "merger/scheme" if hit else "other",
                "haircut": 0.0 if hit else haircut,
            }
        )
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- events
def to_session(
    ts: datetime | pd.Timestamp, sessions: pd.DatetimeIndex, cutoff: time
) -> pd.Timestamp | None:
    """Session on which information released at ``ts`` is first tradeable at the close."""
    t = pd.Timestamp(ts)
    d = pd.Timestamp(t.date())
    if (t.hour, t.minute) >= (cutoff.hour, cutoff.minute):
        d = d + pd.Timedelta(days=1)
    i = int(sessions.searchsorted(d))
    if i < len(sessions):
        return sessions[i]
    while d.weekday() >= 5:  # beyond the calendar (upcoming meeting): next weekday
        d = d + pd.Timedelta(days=1)
    return d


def build_events(res: pd.DataFrame, sessions: pd.DatetimeIndex, cutoff: time) -> pd.DataFrame:
    """Results → one row per (ticker, quarter): first broadcast + event session."""
    cols = ["symbol", "broadcast", "event_date", "period_to"]
    if res.empty or "ticker" not in res:
        return pd.DataFrame(columns=cols)
    r = res.dropna(subset=["ticker", "broadcast"]).copy()
    r["period_to"] = r["period_to"].fillna(pd.to_datetime(r["broadcast"]).dt.date)
    r = r.sort_values("broadcast").drop_duplicates(["ticker", "period_to"], keep="first")
    r["event_date"] = [to_session(b, sessions, cutoff) for b in r["broadcast"]]
    r = (
        r.dropna(subset=["event_date"])
        .drop(columns=["symbol"])
        .rename(columns={"ticker": "symbol"})
    )
    return r[cols].reset_index(drop=True)


MEETING_MATCH_DAYS = (-2, 7)  # a filing within this many days of the meeting = same event


def add_meeting_events(
    ev: pd.DataFrame, mt: pd.DataFrame, sessions: pd.DatetimeIndex
) -> pd.DataFrame:
    """Fill quarters missing from the results feed with results board meetings.

    Since SEBI's integrated filing (2025) many results no longer appear in the
    financial-results feed; the prior-intimated board meeting still does. The results
    are assumed public after that day's close (event = next session) — conservative:
    at worst one session late, never early.
    """
    cols = ["symbol", "broadcast", "event_date", "period_to", "source"]
    ev = ev.assign(source="filing") if not ev.empty else pd.DataFrame(columns=cols)
    if mt.empty or "ticker" not in mt:
        return ev[cols]
    m = mt[mt["is_results"].astype(bool)].dropna(subset=["ticker", "meeting_date"])
    m = m.assign(md=pd.to_datetime(m["meeting_date"])).drop_duplicates(["ticker", "md"])
    last = sessions[-1] if len(sessions) else pd.Timestamp.max
    have: dict[str, np.ndarray] = {
        str(t): pd.to_datetime(g["broadcast"]).dt.normalize().to_numpy()
        for t, g in ev.groupby("symbol")
    }
    lo, hi = (np.timedelta64(d, "D") for d in MEETING_MATCH_DAYS)
    rows = []
    for t, md in zip(m["ticker"], m["md"], strict=False):
        b = have.get(str(t))
        d64 = np.datetime64(md.normalize())
        if b is not None and ((b >= d64 + lo) & (b <= d64 + hi)).any():
            continue
        i = int(sessions.searchsorted(md.normalize(), side="right"))  # next session
        if i >= len(sessions) or sessions[i] > last:
            continue  # meeting not held yet
        rows.append(
            {
                "symbol": t,
                "broadcast": md.normalize() + pd.Timedelta(hours=18),
                "event_date": sessions[i],
                "period_to": pd.NaT,
                "source": "meeting",
            }
        )
    add = pd.DataFrame(rows, columns=cols)
    out = pd.concat([ev[cols], add], ignore_index=True) if len(add) else ev[cols]
    return out.sort_values(["symbol", "event_date"]).reset_index(drop=True)


def build_meetings(mt: pd.DataFrame, sessions: pd.DatetimeIndex, cutoff: time) -> pd.DataFrame:
    """Results board meetings → meeting session + session from which it was known."""
    cols = ["symbol", "meeting_session", "known_session"]
    if mt.empty or "ticker" not in mt:
        return pd.DataFrame(columns=cols)
    m = mt[mt["is_results"].astype(bool)].dropna(subset=["ticker"]).copy()
    m["meeting_session"] = [
        to_session(datetime.combine(pd.Timestamp(d).date(), time(0, 0)), sessions, cutoff)
        for d in m["meeting_date"]
    ]
    m["known_session"] = [
        to_session(k, sessions, cutoff) if pd.notna(k) else ms
        for k, ms in zip(m["intimation"], m["meeting_session"], strict=False)
    ]
    m = m.dropna(subset=["meeting_session", "known_session"])
    m = m.drop(columns=["symbol"]).rename(columns={"ticker": "symbol"})
    return m[cols].drop_duplicates().reset_index(drop=True)


# --------------------------------------------------------------------------- main
def build_all(
    paths: DataPaths, cfg: AppConfig, progress: Callable[[str], None] | None = None
) -> BuildReport:
    """Build every derived table from the cached NSE data."""
    say = progress or (lambda _m: None)
    rep = BuildReport()
    out = derived(paths)
    out.mkdir(parents=True, exist_ok=True)
    bhav = glob_of(paths, "bhav")
    for k in ("bhav", "mto", "surv"):
        d = paths.nse / k
        setattr(rep, f"{k}_sessions", len(list(d.glob("*/*.parquet"))) if d.exists() else 0)
    if bhav is None:
        say("  no bhavcopies cached - run nse_backfill.py")
        return rep
    uni = pd.read_parquet(paths.universe)["symbol"].tolist() if paths.universe.exists() else []
    cal = (
        pd.read_parquet(paths.calendar)["date"]
        if paths.calendar.exists()
        else pd.Series([], dtype="object")
    )
    sessions = sorted({pd.Timestamp(x).date() for x in cal})
    sess_idx = pd.DatetimeIndex(pd.to_datetime(sessions))
    con = duckdb.connect()
    try:
        stats = con.execute(
            f"""SELECT symbol, CAST(min(date) AS DATE) AS first, CAST(max(date) AS DATE) AS last,
                       count(*) AS n, arg_max(series, date) AS series
                FROM read_parquet('{bhav}', union_by_name=true)
                WHERE volume > 0 GROUP BY symbol"""
        ).fetchdf()
        latest = pd.Timestamp(stats["last"].max()).date()
        stats["first"] = pd.to_datetime(stats["first"]).dt.date
        stats["last"] = pd.to_datetime(stats["last"]).dt.date
        eq = paths.eq_store()
        fy_first: dict[str, date] = {}
        if eq.symbols_on_disk():
            ff = con.execute(
                f"SELECT symbol, CAST(min(date) AS DATE) AS f FROM read_parquet('{eq.glob()}') GROUP BY symbol"
            ).fetchdf()
            fy_first = {
                s: pd.Timestamp(f).date() for s, f in zip(ff["symbol"], ff["f"], strict=False)
            }
        sc = paths.nse / "symbol_change.parquet"
        changes = (
            pd.read_parquet(sc) if sc.exists() else pd.DataFrame(columns=["old", "new", "date"])
        )
        smap = symbol_map(stats, uni, fy_first, changes, latest, sessions, cfg)
        write_parquet_frame(smap, out / "symbol_map.parquet")
        rep.nse_symbols = len(stats)
        rep.mapped_listed = int(smap.loc[~smap["delisted"], "ticker"].nunique())
        rep.delisted = int(smap.loc[smap["delisted"], "ticker"].nunique())
        rep.unmapped_trading = len(set(stats["symbol"]) - set(smap["nse_symbol"]))
        say(
            f"  symbols: {rep.mapped_listed} listed matched, {rep.delisted} delisted, {rep.unmapped_trading} other"
        )

        # corporate actions (mapped) + dividends
        ca = read_api(paths, "ca")
        ca_raw = ca.copy()
        if not ca.empty:
            ca["ticker"] = map_frame(ca, smap, "symbol", "ex_date")
            ca = ca.dropna(subset=["ticker"])
            write_parquet_frame(ca, out / "corp_actions.parquet")
            dv = (
                ca[ca["kind"] == "dividend"]
                .groupby(["ticker", "symbol", "ex_date"], as_index=False)["dividend"]
                .sum()
            )
            if not dv.empty:
                con.register("dv", dv)
                y = con.execute(
                    f"""SELECT dv.ticker AS symbol, CAST(dv.ex_date AS DATE) AS ex_date, dv.dividend,
                               b.prev_close, dv.dividend / b.prev_close AS yield
                        FROM dv JOIN read_parquet('{bhav}', union_by_name=true) b
                          ON b.symbol = dv.symbol AND CAST(b.date AS DATE) = CAST(dv.ex_date AS DATE)
                        WHERE b.prev_close > 0"""
                ).fetchdf()
                con.unregister("dv")
                y = y[(y["yield"] > 0) & (y["yield"] < 0.5)]  # > 50% = parse error / special case
                write_parquet_frame(y, out / "dividends.parquet")
                rep.dividends = len(y)

        # delisted candles
        ds = paths.delisted_store()
        if ds.root.exists():
            shutil.rmtree(ds.root)
        adj: list[dict[str, Any]] = []
        iss: list[dict[str, Any]] = []
        if cfg.nse.survivorship.enabled:
            candles = delisted_candles(con, bhav, smap)
            vc = cfg.settings.validation
            tc = TradingCalendar.from_file(cfg.settings.market, paths.calendar)
            ca_ev = ca_adjustments(ca if not ca.empty else pd.DataFrame(), candles)
            by_sym: dict[str, list[date]] = {}
            for e in ca_ev:
                by_sym.setdefault(e["symbol"], []).append(pd.Timestamp(e["ex_date"]).date())
            for t, c in candles.items():
                ds.write(t, c)
                rep.delisted_rows += len(c)
                iss += validate_symbol(c, t, vc, tc, c["date"].iloc[-1], is_equity=True)
                if vc.ca_auto_adjust:
                    for e in detect_adjustments(c, t, vc):
                        ex = pd.Timestamp(e["ex_date"]).date()
                        if all(abs((ex - k).days) > CA_MATCH_DAYS for k in by_sym.get(t, [])):
                            adj.append(e)
            adj += ca_ev
        reasons = delisting_reasons(smap, ca_raw, cfg.backtest.execution.delist_exit_haircut)
        write_parquet_frame(reasons, out / "delisted_reason.parquet")
        rep.delisted_mergers = int((reasons["reason"] == "merger/scheme").sum())
        write_parquet_frame(adjustments_frame(adj), out / "delisted_adjustments.parquet")
        write_parquet_frame(issues_frame(iss), out / "delisted_issues.parquet")
        rep.delisted_adjustments = len(adj)

        # events + meetings
        cutoff = cfg.nse.results_cutoff
        res = read_api(paths, "results")
        ev = pd.DataFrame()
        if not res.empty:
            res["ticker"] = map_frame(res, smap, "symbol", "broadcast")
            ev = build_events(res, sess_idx, cutoff)
        mt = read_api(paths, "meetings")
        up = paths.nse / "meetings_upcoming.parquet"
        if up.exists():
            mt = pd.concat([mt, pd.read_parquet(up)], ignore_index=True)
        if not mt.empty:
            mt["ticker"] = map_frame(mt, smap, "symbol", "meeting_date")
            mtg = build_meetings(mt, sess_idx, cutoff)
            write_parquet_frame(mtg, out / "meetings.parquet")
            rep.meetings = len(mtg)
        ev = add_meeting_events(ev, mt, sess_idx)
        if not ev.empty:
            write_parquet_frame(ev, out / "events.parquet")
            rep.events = len(ev)
            rep.events_from_meetings = int((ev["source"] == "meeting").sum())
    finally:
        con.close()
    return rep
