"""Repositories: per-symbol candle Parquet store, JSON state, DuckDB database.

Writes are atomic (temp file + replace) so an interrupted backfill never leaves a
half-written Parquet file.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quantswing.core.clock import now_ist
from quantswing.core.config import AppConfig, get_config
from quantswing.store.schema import CANDLE_SCHEMA, safe_name


def candles_for_store(candles: pd.DataFrame, symbol: str, fetched_at: datetime) -> pd.DataFrame:
    """Add ``date``/``symbol``/``fetched_at`` to a Fyers candle frame (``CANDLE_COLUMNS``)."""
    df = candles.copy()
    df.insert(0, "date", df["ts"].dt.date)
    df.insert(2, "symbol", symbol)
    df["fetched_at"] = pd.Timestamp(fetched_at)
    return df[list(CANDLE_SCHEMA.names)]


def merge_candles(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Union of ``old`` and ``new`` by ``date``; ``new`` wins on overlap. Sorted by date."""
    if old is None or old.empty:
        out = new
    elif new.empty:
        out = old
    else:
        out = pd.concat([old, new], ignore_index=True)
    return (
        out.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)
    )


class PriceStore:
    """One Parquet file per symbol under ``root``."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, symbol: str) -> Path:
        """File path for ``symbol``."""
        return self.root / f"{safe_name(symbol)}.parquet"

    def exists(self, symbol: str) -> bool:
        """True if a file exists for ``symbol``."""
        return self.path(symbol).exists()

    def read(self, symbol: str) -> pd.DataFrame | None:
        """Stored candles for ``symbol`` or ``None``."""
        p = self.path(symbol)
        if not p.exists():
            return None
        df: pd.DataFrame = pq.read_table(p).to_pandas()
        return df

    def last_date(self, symbol: str) -> date | None:
        """Latest stored session date for ``symbol`` (reads only the ``date`` column)."""
        p = self.path(symbol)
        if not p.exists():
            return None
        col = pq.read_table(p, columns=["date"]).column("date")
        return None if len(col) == 0 else max(col.to_pylist())

    def write(self, symbol: str, df: pd.DataFrame) -> Path:
        """Atomically write ``df`` (must match ``CANDLE_SCHEMA``) for ``symbol``."""
        self.root.mkdir(parents=True, exist_ok=True)
        p = self.path(symbol)
        tmp = p.with_suffix(".tmp")
        table = pa.Table.from_pandas(df, schema=CANDLE_SCHEMA, preserve_index=False)
        pq.write_table(table, tmp)
        tmp.replace(p)
        return p

    def upsert(self, symbol: str, new: pd.DataFrame) -> pd.DataFrame:
        """Merge ``new`` into the stored file and write it back. Returns the merged frame."""
        merged = merge_candles(self.read(symbol), new)
        if not merged.empty:
            self.write(symbol, merged)
        return merged

    def glob(self) -> str:
        """Glob pattern of all files (for DuckDB ``read_parquet``)."""
        return str(self.root / "*.parquet").replace("\\", "/")

    def symbols_on_disk(self) -> list[Path]:
        """All stored files."""
        return sorted(self.root.glob("*.parquet")) if self.root.exists() else []


class JsonState:
    """Small JSON key/value state file (atomic writes)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {}
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))

    def get(self, key: str, default: Any = None) -> Any:
        """Value for ``key``."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any, *, flush: bool = True) -> None:
        """Set ``key`` and (by default) persist."""
        self.data[key] = value
        if flush:
            self.flush()

    def flush(self) -> None:
        """Persist to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=str), encoding="utf-8")
        tmp.replace(self.path)


def write_parquet_frame(df: pd.DataFrame, path: Path) -> None:
    """Atomic write of an arbitrary frame to Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


class DataPaths:
    """Resolved P1 data paths from config."""

    def __init__(self, cfg: AppConfig | None = None) -> None:
        cfg = cfg or get_config()
        p, d = cfg.settings.paths, cfg.settings.data
        raw = p.resolve(p.raw_dir)
        self.eq_dir = raw / d.equity_subdir
        self.idx_dir = raw / d.index_subdir
        self.universe = p.resolve(d.universe_file)
        self.calendar = p.resolve(d.calendar_file)
        self.issues = p.resolve(d.issues_file)
        self.adjustments = p.resolve(d.adjustments_file)
        self.factors = p.resolve(d.factors_dir)
        self.reports = p.resolve(d.reports_dir)
        self.runs = p.resolve(cfg.backtest.tracking.runs_dir)
        self.api_usage = p.resolve(p.api_usage_file)
        self.token = p.resolve(p.token_file)
        self.state = p.resolve(d.backfill_state_file)
        self.duckdb = p.resolve(d.duckdb_file)
        ns = cfg.nse.storage
        self.nse = p.resolve(ns.processed_dir)  # parsed NSE data (D31)
        self.nse_raw = p.resolve(ns.raw_dir)
        self.nse_state = p.resolve(ns.state_file)
        self.paper_db = p.resolve(cfg.paper.db_file)  # P6 operational store (SQLite)
        self.macro_symbols = p.resolve(Path(cfg.factors.macro.symbols_file))  # D39 drivers
        self.live_db = p.resolve(cfg.paper.live.db_file)  # D40 live paper book
        self.live_eod_log = p.resolve(cfg.paper.live.eod_log)
        self.live_poll_log = p.resolve(cfg.paper.live.poll_log)

    @classmethod
    def relocated(cls, root: Path, cfg: AppConfig | None = None) -> DataPaths:
        """All data paths under ``root`` (same layout; used by tests and the dashboard demo)."""
        p = cls(cfg)
        p.eq_dir = root / "raw" / "eq"
        p.idx_dir = root / "raw" / "idx"
        p.universe = root / "processed" / "universe.parquet"
        p.calendar = root / "processed" / "calendar.parquet"
        p.issues = root / "processed" / "issues.parquet"
        p.adjustments = root / "processed" / "adjustments.parquet"
        p.factors = root / "processed" / "factors"
        p.reports = root / "reports"
        p.runs = root / "runs"
        p.state = root / "db" / "state.json"
        p.duckdb = root / "db" / "qs.duckdb"
        p.api_usage = root / "db" / "api_usage.json"
        p.nse = root / "processed" / "nse"
        p.nse_raw = root / "raw" / "nse"
        p.nse_state = root / "db" / "nse_state.json"
        p.paper_db = root / "db" / "paper.sqlite"
        p.macro_symbols = root / "processed" / "macro_symbols.json"
        p.live_db = root / "db" / "live.sqlite"
        p.live_eod_log = root / "logs" / "live_eod.log"
        p.live_poll_log = root / "logs" / "live_poll.log"
        return p

    def eq_store(self) -> PriceStore:
        """Equity candle store."""
        return PriceStore(self.eq_dir)

    def idx_store(self) -> PriceStore:
        """Index candle store."""
        return PriceStore(self.idx_dir)

    def delisted_store(self) -> PriceStore:
        """Delisted-stock candles rebuilt from NSE bhavcopies (survivorship fix, D31)."""
        return PriceStore(self.nse / "delisted_eq")


def connect(path: Path, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the DuckDB database."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def stamp() -> datetime:
    """Fetch timestamp (IST)."""
    return now_ist()


def ensure_list(x: Iterable[str]) -> list[str]:
    """Materialise an iterable of symbols."""
    return list(dict.fromkeys(x))
