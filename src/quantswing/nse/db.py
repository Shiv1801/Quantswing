"""DuckDB tables from NSE data (D31). Every table exists (empty) even without NSE data, so
the rest of the platform never needs to check.

======================  ===============================================================
``nse_symbol_map``      nse_symbol, d_from, d_to, ticker, delisted
``nse_delivery``        symbol (ticker), date, delivery_qty, delivery_pct (0–100)
``surveillance``        date, symbol, restricted, pledge, loss_making (flagged rows only)
``surveillance_dates``  sessions with a surveillance file (coverage)
``corp_actions``        symbol, ex_date, kind, ratio, dividend, subject
``dividends``           symbol, ex_date, dividend, prev_close, yield
``results_events``      symbol, broadcast, event_date, period_to
``results_meetings``    symbol, meeting_session, known_session
``sectors``             symbol, macro, sector, industry, basic_industry
``holidays``            date, description
``delisted``            symbol, first_date, last_date, sessions
======================  ===============================================================
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from quantswing.core.config import AppConfig
from quantswing.nse.build import derived, glob_of
from quantswing.store.repositories import DataPaths

EMPTY: dict[str, str] = {
    "nse_symbol_map": "nse_symbol VARCHAR, d_from DATE, d_to DATE, ticker VARCHAR, delisted BOOLEAN",
    "nse_delivery": "symbol VARCHAR, date DATE, delivery_qty DOUBLE, delivery_pct DOUBLE",
    "surveillance": "date DATE, symbol VARCHAR, restricted BOOLEAN, pledge BOOLEAN, loss_making BOOLEAN",
    "surveillance_dates": "date DATE",
    "corp_actions": "symbol VARCHAR, ex_date DATE, kind VARCHAR, ratio DOUBLE, dividend DOUBLE, subject VARCHAR",
    "dividends": "symbol VARCHAR, ex_date DATE, dividend DOUBLE, prev_close DOUBLE, yield DOUBLE",
    "results_events": "symbol VARCHAR, broadcast TIMESTAMP, event_date DATE, period_to DATE, "
    "source VARCHAR",
    "results_meetings": "symbol VARCHAR, meeting_session DATE, known_session DATE",
    "sectors": "symbol VARCHAR, macro VARCHAR, sector VARCHAR, industry VARCHAR, basic_industry VARCHAR",
    "holidays": "date DATE, description VARCHAR",
    "delisted": "symbol VARCHAR, first_date DATE, last_date DATE, sessions BIGINT, "
    "reason VARCHAR, haircut DOUBLE",
}
SERIES_RANK_SQL = "CASE series WHEN 'EQ' THEN 0 WHEN 'BE' THEN 1 ELSE 2 END"


def _p(path: Path) -> str:
    return str(path).replace("\\", "/")


def _empty(con: duckdb.DuckDBPyConnection, name: str) -> None:
    con.execute(f"CREATE OR REPLACE TABLE {name} ({EMPTY[name]})")


def _from_parquet(con: duckdb.DuckDBPyConnection, name: str, path: Path, select: str) -> None:
    if path.exists():
        con.execute(
            f"CREATE OR REPLACE TABLE {name} AS SELECT {select} FROM read_parquet('{_p(path)}')"
        )
    else:
        _empty(con, name)


def build_nse_tables(con: duckdb.DuckDBPyConnection, paths: DataPaths, cfg: AppConfig) -> None:
    """(Re)create every NSE table in ``con``."""
    d = derived(paths)
    _from_parquet(
        con,
        "nse_symbol_map",
        d / "symbol_map.parquet",
        "nse_symbol, CAST(d_from AS DATE) AS d_from, CAST(d_to AS DATE) AS d_to, ticker, delisted",
    )
    mto = glob_of(paths, "mto")
    if mto:
        con.execute(f"""CREATE OR REPLACE TABLE nse_delivery AS
                SELECT m.ticker AS symbol, CAST(x.date AS DATE) AS date,
                       arg_min(x.deliv_qty, {SERIES_RANK_SQL}) AS delivery_qty,
                       arg_min(x.deliv_pct, {SERIES_RANK_SQL}) AS delivery_pct
                FROM read_parquet('{mto}', union_by_name=true) x
                JOIN nse_symbol_map m ON x.symbol = m.nse_symbol
                 AND CAST(x.date AS DATE) BETWEEN m.d_from AND m.d_to
                GROUP BY 1, 2""")
    else:
        _empty(con, "nse_delivery")
    surv = glob_of(paths, "surv")
    if surv:
        flags = cfg.nse.surveillance_exclude
        restricted = " OR ".join(f'x."{f}"' for f in flags) if flags else "FALSE"
        con.execute(f"""CREATE OR REPLACE TABLE surveillance AS
                SELECT CAST(x.date AS DATE) AS date, m.ticker AS symbol,
                       bool_or({restricted}) AS restricted,
                       bool_or(x.pledge OR x.total_pledge OR x.encumbered) AS pledge,
                       bool_or(x.loss_making) AS loss_making
                FROM read_parquet('{surv}', union_by_name=true) x
                JOIN nse_symbol_map m ON x.symbol = m.nse_symbol
                 AND CAST(x.date AS DATE) BETWEEN m.d_from AND m.d_to
                GROUP BY 1, 2
                HAVING bool_or({restricted}) OR bool_or(x.pledge OR x.total_pledge OR x.encumbered)
                    OR bool_or(x.loss_making)""")
        con.execute(
            f"""CREATE OR REPLACE TABLE surveillance_dates AS
                SELECT DISTINCT CAST(date AS DATE) AS date FROM read_parquet('{surv}', union_by_name=true)"""
        )
    else:
        _empty(con, "surveillance")
        _empty(con, "surveillance_dates")
    _from_parquet(
        con,
        "corp_actions",
        d / "corp_actions.parquet",
        "ticker AS symbol, CAST(ex_date AS DATE) AS ex_date, kind, ratio, dividend, subject",
    )
    _from_parquet(
        con,
        "dividends",
        d / "dividends.parquet",
        "symbol, CAST(ex_date AS DATE) AS ex_date, dividend, prev_close, yield",
    )
    _from_parquet(
        con,
        "results_events",
        d / "events.parquet",
        "symbol, CAST(broadcast AS TIMESTAMP) AS broadcast, CAST(event_date AS DATE) AS event_date, "
        "CAST(period_to AS DATE) AS period_to, source",
    )
    _from_parquet(
        con,
        "results_meetings",
        d / "meetings.parquet",
        "symbol, CAST(meeting_session AS DATE) AS meeting_session, "
        "CAST(known_session AS DATE) AS known_session",
    )
    _from_parquet(
        con,
        "sectors",
        paths.nse / "sectors.parquet",
        "symbol, macro, sector, industry, basic_industry",
    )
    _from_parquet(
        con,
        "holidays",
        paths.nse / "holidays.parquet",
        "CAST(date AS DATE) AS date, description",
    )
    ds = paths.delisted_store()
    if ds.symbols_on_disk():
        rs = d / "delisted_reason.parquet"
        join = (
            f"LEFT JOIN read_parquet('{_p(rs)}') r USING (symbol)"
            if rs.exists()
            else "LEFT JOIN (SELECT NULL::VARCHAR AS symbol, NULL::VARCHAR AS reason, "
            "NULL::DOUBLE AS haircut) r USING (symbol)"
        )
        con.execute(f"""CREATE OR REPLACE TABLE delisted AS
                SELECT g.symbol, g.first_date, g.last_date, g.sessions, r.reason, r.haircut
                FROM (SELECT symbol, min(date) AS first_date, max(date) AS last_date,
                             count(*) AS sessions
                      FROM read_parquet('{ds.glob()}') GROUP BY symbol) g {join}""")
    else:
        _empty(con, "delisted")


def ensure_empty_nse_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create missing NSE tables empty (e.g. NSE disabled)."""
    have = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    for name in EMPTY:
        if name not in have:
            _empty(con, name)
