"""Wide-panel I/O for P2: load adjusted price panels from DuckDB, persist factor panels.

Panels are indexed by the observed session calendar (so a missing candle is an explicit
NaN) and use split/bonus-adjusted prices (``adj_*`` columns, P1+).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from quantswing.core.config import AppConfig, get_config
from quantswing.core.logging import get_logger
from quantswing.factors.common import Panels
from quantswing.store.repositories import DataPaths, connect, write_parquet_frame

log = get_logger(__name__)

PRICE_FIELDS = {
    "open": "adj_open",
    "high": "adj_high",
    "low": "adj_low",
    "close": "adj_close",
    "volume": "adj_volume",
    "turnover": "turnover_est",
}


def load_panels(
    paths: DataPaths,
    cfg: AppConfig | None = None,
    symbols: Sequence[str] | None = None,
    limit: int | None = None,
) -> Panels:
    """Adjusted OHLCV + turnover panels and the benchmark close, on the session calendar."""
    cfg = cfg or get_config()
    con = connect(paths.duckdb, read_only=True)
    try:
        cal = con.execute("SELECT date FROM calendar ORDER BY date").fetchdf()["date"]
        syms = (
            list(symbols)
            if symbols
            else [
                r[0]
                for r in con.execute(
                    "SELECT DISTINCT symbol FROM prices_daily ORDER BY symbol"
                ).fetchall()
            ]
        )
        if limit:
            syms = syms[:limit]
        con.register("sel", pd.DataFrame({"symbol": syms}))
        cols = ", ".join(f"{src} AS {dst}" for dst, src in PRICE_FIELDS.items())
        long = con.execute(
            f"SELECT date, symbol, {cols} FROM prices_daily JOIN sel USING (symbol)"
        ).fetchdf()
        bench_sym = cfg.factors.beta.benchmark
        bench = con.execute(
            "SELECT date, close FROM indices_daily WHERE symbol = ? ORDER BY date", [bench_sym]
        ).fetchdf()
        idx = pd.DatetimeIndex(pd.to_datetime(cal), name="date")
        extras = _nse_extras(con, idx, syms) if cfg.nse.enabled else {}
    finally:
        con.close()
    if bench.empty:
        raise RuntimeError(f"benchmark {bench_sym} has no data in indices_daily")
    long["date"] = pd.to_datetime(long["date"])
    panels = {
        f: long.pivot(index="date", columns="symbol", values=f)
        .reindex(index=idx, columns=syms)
        .astype("float64")
        for f in PRICE_FIELDS
    }
    b = bench.assign(date=pd.to_datetime(bench["date"])).set_index("date")["close"].reindex(idx)
    log.info("panels loaded", extra={"sessions": len(idx), "symbols": len(syms)})
    kw: dict[str, Any] = {**panels, **extras}
    if cfg.factors.macro.enabled:
        kw["macro"] = load_macro(paths, cfg, idx)
    return Panels(bench=b.astype("float64"), **kw)


def macro_symbols(paths: DataPaths, cfg: AppConfig) -> dict[str, str]:
    """Driver name → Fyers symbol: the base indices plus what ``macro_backfill.py`` found."""
    out = dict(cfg.factors.macro.base)
    if paths.macro_symbols.exists():
        extra = json.loads(paths.macro_symbols.read_text(encoding="utf-8"))
        out.update({str(k): str(v) for k, v in extra.items()})
    return out


def load_macro(paths: DataPaths, cfg: AppConfig, idx: pd.DatetimeIndex) -> pd.DataFrame | None:
    """D39: driver closes on the session calendar (short gaps forward-filled, max 3 sessions)."""
    store = paths.idx_store()
    cols: dict[str, pd.Series] = {}
    for name, sym in macro_symbols(paths, cfg).items():
        df = store.read(sym)
        if df is None or df.empty or "close" not in df:
            continue
        s = df.assign(date=pd.to_datetime(df["date"])).set_index("date")["close"]
        s = s[~s.index.duplicated(keep="last")].astype("float64")
        cols[name] = s.reindex(idx).ffill(limit=3)
    return pd.DataFrame(cols, index=idx) if cols else None


def _query(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame | None:
    """Result frame, or ``None`` if the table does not exist / is empty (no NSE data)."""
    try:
        df = con.execute(sql).fetchdf()
    except (duckdb.CatalogException, duckdb.BinderException):
        return None
    return df if not df.empty else None


def _wide(long: pd.DataFrame, col: str, idx: pd.DatetimeIndex, syms: list[str]) -> pd.DataFrame:
    x = long.assign(date=pd.to_datetime(long["date"]))
    w = x.pivot_table(index="date", columns="symbol", values=col, aggfunc="sum")
    return w.reindex(index=idx, columns=syms).astype("float64")


def _nse_extras(
    con: duckdb.DuckDBPyConnection, idx: pd.DatetimeIndex, syms: list[str]
) -> dict[str, Any]:
    """Optional NSE panels (D31) for :class:`Panels`; absent tables → field omitted."""
    out: dict[str, Any] = {}
    sel = "JOIN sel USING (symbol)"
    dv = _query(
        con,
        f"SELECT date, symbol, delivery_pct FROM prices_daily {sel} WHERE delivery_pct IS NOT NULL",
    )
    if dv is not None:
        out["deliv"] = _wide(dv, "delivery_pct", idx, syms)
    sec = _query(con, f"SELECT symbol, sector FROM sectors {sel}")
    if sec is not None:
        out["sector"] = sec.drop_duplicates("symbol").set_index("symbol")["sector"].reindex(syms)
    cov = _query(con, "SELECT date FROM surveillance_dates")
    if cov is not None:
        covered = pd.Series(idx.isin(pd.to_datetime(cov["date"])), index=idx)
        sv = _query(con, f"SELECT date, symbol, restricted, pledge FROM surveillance {sel}")
        base = pd.DataFrame(0.0, index=idx, columns=syms)
        restricted, pledge = base.copy(), base.copy()
        if sv is not None:
            restricted = _wide(sv.assign(v=sv["restricted"].astype(float)), "v", idx, syms).fillna(
                0.0
            )
            pledge = _wide(sv.assign(v=sv["pledge"].astype(float)), "v", idx, syms).fillna(0.0)
        out["restricted"] = restricted.gt(0)
        out["pledge"] = pledge.clip(upper=1.0).where(covered, axis=0)
    ev = _query(con, f"SELECT symbol, event_date FROM results_events {sel}")
    if ev is not None:
        out["events"] = ev.assign(event_date=pd.to_datetime(ev["event_date"]))
    mt = _query(con, f"SELECT symbol, meeting_session, known_session FROM results_meetings {sel}")
    if mt is not None:
        out["meetings"] = mt.assign(
            meeting_session=pd.to_datetime(mt["meeting_session"]),
            known_session=pd.to_datetime(mt["known_session"]),
        )
    dy = _query(con, f"SELECT ex_date AS date, symbol, yield FROM dividends {sel}")
    if dy is not None:
        out["div_yield"] = _wide(dy, "yield", idx, syms).fillna(0.0)
    dl = _query(con, f"SELECT symbol, last_date, haircut FROM delisted {sel}")
    if dl is None:  # database built before per-symbol haircuts existed
        dl = _query(con, f"SELECT symbol, last_date, NULL AS haircut FROM delisted {sel}")
    if dl is not None:
        out["delisted_last"] = pd.Series(
            pd.to_datetime(dl["last_date"]).to_numpy(), index=dl["symbol"]
        )
        out["delisted_haircut"] = pd.Series(
            pd.to_numeric(dl["haircut"], errors="coerce").to_numpy(), index=dl["symbol"]
        )
    return out


class FactorStore:
    """Wide Parquet per factor under ``root/{kind}/{name}.parquet`` (kind: raw|z|neutral)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, kind: str, name: str) -> Path:
        """File path for a factor panel."""
        return self.root / kind / f"{name}.parquet"

    def write(self, kind: str, name: str, df: pd.DataFrame) -> None:
        """Persist a panel as float32 (dates as a column)."""
        out = df.astype("float32").copy()
        out.index.name = "date"
        write_parquet_frame(out.reset_index(), self.path(kind, name))

    def read(self, kind: str, name: str) -> pd.DataFrame:
        """Load a panel (float64)."""
        df = pd.read_parquet(self.path(kind, name))
        return df.set_index(pd.DatetimeIndex(df.pop("date"), name="date")).astype("float64")

    def names(self, kind: str) -> list[str]:
        """Stored factor names for ``kind``."""
        d = self.root / kind
        return sorted(p.stem for p in d.glob("*.parquet")) if d.exists() else []
