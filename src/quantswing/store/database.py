"""Build the analytical DuckDB database from the Parquet stores.

Tables: ``universe``, ``calendar``, ``validation_issues``, ``raw_eq`` (view),
``prices_daily`` (validated equities, reject rows removed, quarantine flagged),
``indices_daily`` + the NSE tables of :mod:`quantswing.nse.db` (D31).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from quantswing.core.config import AppConfig, get_config
from quantswing.core.logging import get_logger
from quantswing.nse.build import derived
from quantswing.nse.db import build_nse_tables, ensure_empty_nse_tables
from quantswing.store.repositories import DataPaths, PriceStore, connect, write_parquet_frame
from quantswing.store.schema import DUCKDB_DDL, ISSUE_COLUMNS, UNIVERSE_COLUMNS
from quantswing.store.validation import adjustments_frame

log = get_logger(__name__)


def build_calendar(idx_store: PriceStore, calendar_symbol: str, paths: DataPaths) -> int:
    """Write the observed-session calendar from the calendar index. Returns #sessions."""
    df = idx_store.read(calendar_symbol)
    if df is None or df.empty:
        raise RuntimeError(f"calendar index {calendar_symbol} has no data — load indices first")
    cal = pd.DataFrame({"date": sorted(set(df["date"]))})
    write_parquet_frame(cal, paths.calendar)
    return len(cal)


def _empty(cols: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series([], dtype="object") for c in cols})


def _read_or(path: Path, empty: pd.DataFrame) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else empty


def build_database(paths: DataPaths, cfg: AppConfig | None = None) -> dict[str, int]:
    """(Re)build DuckDB tables. Returns row counts per table.

    With NSE data (D31): delisted-stock candles join ``raw_eq`` (survivorship fix), their
    adjustments/issues join the Fyers ones, and delivery % fills ``prices_daily``.
    """
    cfg = cfg or get_config()
    con = connect(paths.duckdb)
    nd = derived(paths)
    try:
        uni = (
            pd.read_parquet(paths.universe) if paths.universe.exists() else _empty(UNIVERSE_COLUMNS)
        )
        cal = pd.read_parquet(paths.calendar) if paths.calendar.exists() else _empty(("date",))
        iss = pd.concat(
            [
                _read_or(paths.issues, _empty(ISSUE_COLUMNS)),
                _read_or(nd / "delisted_issues.parquet", _empty(ISSUE_COLUMNS)),
            ],
            ignore_index=True,
        )
        adj = pd.concat(
            [
                _read_or(paths.adjustments, adjustments_frame([])),
                _read_or(nd / "delisted_adjustments.parquet", adjustments_frame([])),
            ],
            ignore_index=True,
        )
        con.register("adj_df", adj)
        con.execute(
            "CREATE OR REPLACE TABLE adj_events AS SELECT symbol, CAST(ex_date AS DATE) AS ex_date, "
            "CAST(ratio AS DOUBLE) AS ratio, CAST(observed AS DOUBLE) AS observed FROM adj_df"
        )
        con.unregister("adj_df")
        con.register("uni_df", uni)
        con.register("cal_df", cal)
        con.register("iss_df", iss)
        con.execute("CREATE OR REPLACE TABLE universe AS SELECT * FROM uni_df")
        con.execute(
            "CREATE OR REPLACE TABLE calendar AS SELECT CAST(date AS DATE) AS date FROM cal_df"
        )
        con.execute(
            "CREATE OR REPLACE TABLE validation_issues AS "
            'SELECT symbol, CAST(date AS DATE) AS date, "check", severity, detail FROM iss_df'
        )
        for tmp in ("uni_df", "cal_df", "iss_df"):
            con.unregister(tmp)
        if cfg.nse.enabled:
            build_nse_tables(con, paths, cfg)
        ensure_empty_nse_tables(con)
        eq, ix, dl = paths.eq_store(), paths.idx_store(), paths.delisted_store()
        if eq.symbols_on_disk():
            src = [f"SELECT * FROM read_parquet('{eq.glob()}')"]
            if cfg.nse.enabled and cfg.nse.survivorship.enabled and dl.symbols_on_disk():
                src.append(f"SELECT * FROM read_parquet('{dl.glob()}')")
            con.execute(f"CREATE OR REPLACE VIEW raw_eq AS {' UNION ALL '.join(src)}")
            con.execute(DUCKDB_DDL)
        if ix.symbols_on_disk():
            con.execute(
                "CREATE OR REPLACE TABLE indices_daily AS "
                f"SELECT date, symbol, open, high, low, close, volume FROM read_parquet('{ix.glob()}')"
            )
        names = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        counts = {
            n: int(con.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0])  # type: ignore[index]
            for n in names
            if n != "raw_eq"
        }
    finally:
        con.close()
    log.info("duckdb built", extra={"tables": counts})
    return counts


def read_dates(path: str) -> list[object]:
    """Helper: ``date`` column of a Parquet file as Python objects."""
    return list(pq.read_table(path, columns=["date"]).column("date").to_pylist())
