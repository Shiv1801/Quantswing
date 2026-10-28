"""Storage schemas (02 §4–5, adapted for D22 Fyers-only data).

Layout
------
* ``data/raw/fyers/eq_daily/<SYMBOL>.parquet`` — one file per equity, candles exactly as
  Fyers returned them (+ ``date``, ``symbol``, ``fetched_at``). Per-symbol partitioning
  replaces 02's per-date partitioning because Fyers history is fetched per symbol.
* ``data/raw/fyers/idx_daily/<SYMBOL>.parquet`` — same, for indices.
* ``data/db/quantswing.duckdb`` — analytical DB: ``universe``, ``calendar``,
  ``prices_daily`` (clean, validated), ``indices_daily``, ``validation_issues``.
"""

from __future__ import annotations

from typing import Final

import pyarrow as pa

#: Per-symbol candle file schema.
CANDLE_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("date", pa.date32(), nullable=False),
        pa.field("ts", pa.timestamp("ns", tz="Asia/Kolkata"), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.int64()),
        pa.field("fetched_at", pa.timestamp("ns", tz="Asia/Kolkata"), nullable=False),
    ]
)

#: Universe file columns (from the Fyers symbol master).
UNIVERSE_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "name",
    "isin",
    "series",
    "instrument_type",
    "lot_size",
    "tick_size",
    "fytoken",
    "as_of_date",
)

#: Validation issue columns.
ISSUE_COLUMNS: Final[tuple[str, ...]] = ("symbol", "date", "check", "severity", "detail")

#: Severities: ``reject`` rows are excluded from ``prices_daily``; ``quarantine`` rows are
#: kept but flagged; ``info`` is informational.
SEVERITIES: Final[tuple[str, ...]] = ("reject", "quarantine", "warn", "info")

DUCKDB_DDL: Final[str] = """
CREATE OR REPLACE TABLE prices_daily AS
WITH f AS (
    SELECT r.symbol, r.date, COALESCE(exp(sum(ln(e.ratio))), 1.0) AS adj_factor
    FROM raw_eq r
    LEFT JOIN adj_events e ON e.symbol = r.symbol AND e.ex_date > r.date
    GROUP BY r.symbol, r.date
)
SELECT r.date, r.symbol, u.isin, r.open, r.high, r.low, r.close, r.volume,
       f.adj_factor,                                               -- own split/bonus factor
       r.open  * f.adj_factor                   AS adj_open,
       r.high  * f.adj_factor                   AS adj_high,
       r.low   * f.adj_factor                   AS adj_low,
       r.close * f.adj_factor                   AS adj_close,
       r.volume / f.adj_factor                  AS adj_volume,
       r.close * r.volume                       AS turnover_est,   -- Fyers gives no turnover
       dl.delivery_qty / f.adj_factor           AS delivery_qty,   -- NSE MTO (D31)
       dl.delivery_pct                          AS delivery_pct,   -- 0–100
       CAST(r.fetched_at AS DATE)               AS as_of_date,
       (q.symbol IS NOT NULL)                   AS quarantined,
       (d.symbol IS NOT NULL)                   AS delisted        -- NSE bhavcopy history (D31)
FROM raw_eq r
JOIN f USING (symbol, date)
LEFT JOIN universe u USING (symbol)
LEFT JOIN nse_delivery dl ON dl.symbol = r.symbol AND dl.date = r.date
LEFT JOIN delisted d ON d.symbol = r.symbol
LEFT JOIN (SELECT DISTINCT symbol, date FROM validation_issues WHERE severity = 'quarantine') q
       USING (symbol, date)
WHERE NOT EXISTS (
    SELECT 1 FROM validation_issues v
    WHERE v.severity = 'reject' AND v.symbol = r.symbol AND v.date = r.date
);
"""


def safe_name(symbol: str) -> str:
    """``NSE:RELIANCE-EQ`` → ``NSE_RELIANCE-EQ`` (Windows-safe file stem)."""
    return symbol.replace(":", "_").replace("/", "_").replace("&", "and")
