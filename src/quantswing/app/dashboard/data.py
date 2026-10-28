"""Read-only data access for the dashboard (no Streamlit imports → unit-testable).

Rules: never hold a DuckDB connection open (the backfill needs the write lock to rebuild
the database) — each query opens read-only, reads, closes; a locked/missing database
returns an empty frame. Everything returns plain pandas objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from quantswing.alpha.research import COMPOSITE, SNAPSHOT_FILE
from quantswing.backtest.tracking import RunTracker
from quantswing.core.clock import IST, today_ist
from quantswing.store.panels import FactorStore
from quantswing.store.repositories import DataPaths


def query(paths: DataPaths, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
    """Run a read-only query; empty frame if the DB is missing or locked."""
    if not paths.duckdb.exists():
        return pd.DataFrame()
    try:
        con = duckdb.connect(str(paths.duckdb), read_only=True)
    except duckdb.Error:
        return pd.DataFrame()
    try:
        return con.execute(sql, params or []).fetchdf()
    except duckdb.Error:
        return pd.DataFrame()
    finally:
        con.close()


def mtime(path: Path) -> datetime | None:
    """File modification time (IST) or None."""
    return datetime.fromtimestamp(path.stat().st_mtime, tz=IST) if path.exists() else None


def _json(path: Path) -> dict[str, Any]:
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data
    except (OSError, ValueError):
        return {}


# ----------------------------------------------------------------------------- status
@dataclass(frozen=True)
class Status:
    """Top-line system status."""

    universe: int
    eq_done: int
    eq_failed: int
    idx_done: int
    last_update: str | None
    sessions: int
    first_session: date | None
    last_session: date | None
    db_built: datetime | None
    factors_built: datetime | None
    api_today: int
    api_limit: int
    token_file: datetime | None


def backfill_state(paths: DataPaths) -> pd.DataFrame:
    """One row per symbol from ``backfill_state.json`` (readable while the backfill runs)."""
    st = _json(paths.state)
    rows = []
    for key, v in st.items():
        kind, _, sym = key.partition(":")
        rows.append({"kind": kind, "symbol": sym, **(v if isinstance(v, dict) else {})})
    return pd.DataFrame(rows)


def status(paths: DataPaths) -> Status:
    """Collect status from state files (no DB needed)."""
    bs = backfill_state(paths)
    uni = len(pd.read_parquet(paths.universe, columns=["symbol"])) if paths.universe.exists() else 0
    eq = bs[bs["kind"] == "eq"] if not bs.empty else bs
    done = int(eq["complete"].fillna(False).astype(bool).sum()) if "complete" in eq else 0
    failed = int(eq["error"].notna().sum()) if "error" in eq else 0
    idx_done = int((bs["kind"] == "idx").sum()) if not bs.empty else 0
    last = (
        str(bs["updated"].dropna().max())
        if "updated" in bs and bs["updated"].notna().any()
        else None
    )
    sessions: list[date] = []
    if paths.calendar.exists():
        sessions = pq.read_table(paths.calendar, columns=["date"]).column("date").to_pylist()
    usage = _json(paths.api_usage)
    today_count = int(usage.get("count", 0)) if usage.get("day") == today_ist().isoformat() else 0
    return Status(
        universe=uni,
        eq_done=done,
        eq_failed=failed,
        idx_done=idx_done,
        last_update=last,
        sessions=len(sessions),
        first_session=min(sessions) if sessions else None,
        last_session=max(sessions) if sessions else None,
        db_built=mtime(paths.duckdb),
        factors_built=mtime(paths.factors / SNAPSHOT_FILE),
        api_today=today_count,
        api_limit=0,
        token_file=mtime(paths.token),
    )


# ----------------------------------------------------------------------------- data health
def issues(paths: DataPaths) -> pd.DataFrame:
    """Validation issues (empty if not yet validated)."""
    return pd.read_parquet(paths.issues) if paths.issues.exists() else pd.DataFrame()


def adjustments(paths: DataPaths) -> pd.DataFrame:
    """Auto split/bonus adjustments."""
    return pd.read_parquet(paths.adjustments) if paths.adjustments.exists() else pd.DataFrame()


# ----------------------------------------------------------------------------- scanner
def snapshot(paths: DataPaths) -> pd.DataFrame:
    """Latest-sessions factor snapshot incl. composite (long: date, symbol, factor, raw, neutral)."""
    parts = []
    for f in (paths.factors / SNAPSHOT_FILE, paths.factors / "composite_snapshot.parquet"):
        if f.exists():
            parts.append(pd.read_parquet(f))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def scanner_table(snap: pd.DataFrame, on: pd.Timestamp | None = None) -> pd.DataFrame:
    """Wide table for one session: rows = symbols, columns = neutral factor z + composite + gate.

    Sorted by composite (desc). ``on`` defaults to the latest session in the snapshot.
    """
    if snap.empty:
        return pd.DataFrame()
    day = on if on is not None else snap["date"].max()
    s = snap[snap["date"] == day]
    wide = s.pivot_table(index="symbol", columns="factor", values="neutral", aggfunc="first")
    if "GATE" in wide:
        wide["GATE"] = wide["GATE"].fillna(0).astype(bool)
    sort_col = COMPOSITE if COMPOSITE in wide else wide.columns[0]
    first = [c for c in (COMPOSITE, "GATE") if c in wide]
    wide = wide[first + [c for c in wide.columns if c not in first]]
    return wide.sort_values(sort_col, ascending=False)


def universe(paths: DataPaths) -> pd.DataFrame:
    """Universe file."""
    return pd.read_parquet(paths.universe) if paths.universe.exists() else pd.DataFrame()


# ----------------------------------------------------------------------------- stock
def price_history(paths: DataPaths, symbol: str) -> pd.DataFrame:
    """Adjusted daily prices for one symbol."""
    df = query(
        paths,
        "SELECT date, adj_open AS open, adj_high AS high, adj_low AS low, adj_close AS close, "
        "adj_volume AS volume, turnover_est, quarantined FROM prices_daily "
        "WHERE symbol = ? ORDER BY date",
        [symbol],
    )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def factor_history(
    paths: DataPaths, symbol: str, factors: list[str], kind: str = "neutral"
) -> pd.DataFrame:
    """Time series of selected factors for one symbol (reads one column per file)."""
    store = FactorStore(paths.factors)
    out: dict[str, pd.Series] = {}
    for f in factors:
        k = "composite" if f == COMPOSITE else kind
        path = store.path(k, f)
        if not path.exists():
            continue
        cols = pq.read_schema(path).names
        if symbol not in cols:
            continue
        t = pq.read_table(path, columns=["date", symbol]).to_pandas()
        out[f] = t.set_index(pd.to_datetime(t["date"]))[symbol]
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- IC
def ic_results(paths: DataPaths) -> dict[str, pd.DataFrame]:
    """P2 CSV outputs (stats, by_year, verdicts) if present."""
    out: dict[str, pd.DataFrame] = {}
    for key, name in (
        ("stats", "p2_ic_stats.csv"),
        ("by_year", "p2_ic_by_year.csv"),
        ("verdicts", "p2_ic_verdicts.csv"),
    ):
        f = paths.reports / name
        if f.exists():
            out[key] = pd.read_csv(f, index_col=0 if key != "stats" else None)
    return out


# ----------------------------------------------------------------------------- backtests
def runs(paths: DataPaths) -> pd.DataFrame:
    """All logged runs (newest first)."""
    df = RunTracker(paths.runs).runs()
    return df.sort_values("run_id", ascending=False).reset_index(drop=True) if not df.empty else df


def run_artifact(paths: DataPaths, run_id: str, name: str) -> pd.DataFrame:
    """A run's ``trades`` or ``daily`` artifact."""
    f = paths.runs / run_id / f"{name}.parquet"
    return pd.read_parquet(f) if f.exists() else pd.DataFrame()


def benchmark(paths: DataPaths, symbol: str) -> pd.Series:
    """Index close series."""
    df = query(
        paths, "SELECT date, close FROM indices_daily WHERE symbol = ? ORDER BY date", [symbol]
    )
    if df.empty:
        return pd.Series(dtype="float64")
    return df.set_index(pd.to_datetime(df["date"]))["close"]
