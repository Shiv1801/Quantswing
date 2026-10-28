"""P1 data pipeline (D22: Fyers-only).

Steps (each callable on its own; scripts compose them):
1. :func:`refresh_universe` — Fyers symbol master → ``universe.parquet``.
2. :func:`load_indices` — index daily history (calendar source + context/benchmark).
3. :func:`build_calendar` — observed sessions from the calendar index.
4. :func:`load_equities` — equity daily history (resumable).
5. :func:`validate_all` — per-symbol checks → ``validation_issues.parquet``.
6. :func:`build_database` — DuckDB tables.
7. :func:`exit_report` — P1 exit-test metrics.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd
import requests

from quantswing.broker.fyers_data import FyersData
from quantswing.core.calendar import TradingCalendar
from quantswing.core.config import AppConfig, get_config
from quantswing.core.logging import get_logger
from quantswing.ingest.fyers_history import HistoryLoader, RunSummary
from quantswing.ingest.fyers_universe import download_symbol_master, parse_symbol_master
from quantswing.ingest.macro import load_symbols
from quantswing.store import database
from quantswing.store.repositories import DataPaths, JsonState, write_parquet_frame
from quantswing.store.validation import (
    EventProbe,
    adjustments_frame,
    detect_adjustments,
    issues_frame,
    probe_known_event,
    validate_symbol,
)

log = get_logger(__name__)
Progress = Callable[[int, int, RunSummary], None]


def refresh_universe(
    paths: DataPaths, cfg: AppConfig | None = None, http: Any = requests
) -> pd.DataFrame:
    """Download + filter the symbol master; write ``universe.parquet``."""
    cfg = cfg or get_config()
    uc = cfg.settings.data.universe
    uni = parse_symbol_master(download_symbol_master(uc, http), uc)
    write_parquet_frame(uni, paths.universe)
    log.info(
        "universe refreshed", extra={"symbols": len(uni), "types": uni.attrs.get("type_counts")}
    )
    return uni


def load_universe(paths: DataPaths) -> pd.DataFrame:
    """Read ``universe.parquet``."""
    if not paths.universe.exists():
        raise FileNotFoundError("universe.parquet missing — run the universe step first")
    return pd.read_parquet(paths.universe)


def load_indices(
    fd: FyersData, paths: DataPaths, start: date, end: date, cfg: AppConfig | None = None
) -> RunSummary:
    """Load all configured indices."""
    cfg = cfg or get_config()
    state = JsonState(paths.state)
    loader = HistoryLoader(fd, paths.idx_store(), state, kind="idx", cfg=cfg)
    syms = list(cfg.settings.data.indices)
    syms += [t for t in load_symbols(paths.macro_symbols).values() if t not in syms]  # D39
    return loader.run(syms, start, end)


def build_calendar(paths: DataPaths, cfg: AppConfig | None = None) -> int:
    """Observed-session calendar from the calendar index."""
    cfg = cfg or get_config()
    return database.build_calendar(paths.idx_store(), cfg.settings.data.calendar_symbol, paths)


def load_equities(
    fd: FyersData,
    paths: DataPaths,
    symbols: Sequence[str],
    start: date,
    end: date,
    progress: Progress | None = None,
    cfg: AppConfig | None = None,
) -> RunSummary:
    """Load equity history for ``symbols`` (resumable)."""
    cfg = cfg or get_config()
    state = JsonState(paths.state)
    loader = HistoryLoader(fd, paths.eq_store(), state, kind="eq", cfg=cfg)
    return loader.run(symbols, start, end, progress)


def validate_all(
    paths: DataPaths, symbols: Sequence[str], latest: date, cfg: AppConfig | None = None
) -> pd.DataFrame:
    """Validate every stored equity in ``symbols`` + indices; write issues parquet."""
    cfg = cfg or get_config()
    vc = cfg.settings.validation
    cal = TradingCalendar.from_file(cfg.settings.market, paths.calendar)
    issues: list[dict[str, Any]] = []
    adjustments: list[dict[str, Any]] = []
    eq = paths.eq_store()
    for sym in symbols:
        df = eq.read(sym)
        if df is not None and not df.empty:
            issues += validate_symbol(df, sym, vc, cal, latest, is_equity=True)
            if vc.ca_auto_adjust:
                adjustments += detect_adjustments(df, sym, vc)
    ix = paths.idx_store()
    for sym in cfg.settings.data.indices:
        df = ix.read(sym)
        if df is not None and not df.empty:
            issues += validate_symbol(df, sym, vc, cal, latest, is_equity=False)
    out = issues_frame(issues)
    write_parquet_frame(out, paths.issues)
    write_parquet_frame(adjustments_frame(adjustments), paths.adjustments)
    return out


@dataclass
class ExitReport:
    """P1 exit-test metrics and verdict."""

    universe: int = 0
    with_data: int = 0
    coverage: float = 0.0
    rows: int = 0
    rejected_rows: int = 0
    reject_frac: float = 0.0
    quarantined_rows: int = 0
    adjustments: int = 0
    earliest: date | None = None
    latest_session: date | None = None
    fresh_frac: float = 0.0
    sessions: int = 0
    calendar_first: date | None = None
    issue_counts: dict[str, int] = field(default_factory=dict)
    probes: list[EventProbe] = field(default_factory=list)
    earliest_by_year: dict[int, int] = field(default_factory=dict)
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """All exit checks true."""
        return bool(self.checks) and all(self.checks.values())


def exit_report(paths: DataPaths, latest: date, cfg: AppConfig | None = None) -> ExitReport:
    """Compute P1 exit-test metrics from the stores (no network)."""
    cfg = cfg or get_config()
    vc = cfg.settings.validation
    rep = ExitReport(latest_session=latest)
    uni = load_universe(paths)
    rep.universe = len(uni)
    eq = paths.eq_store()
    firsts: list[date] = []
    fresh = 0
    for sym in uni["symbol"]:
        last = eq.last_date(sym)
        if last is None:
            continue
        rep.with_data += 1
        df = eq.read(sym)
        assert df is not None
        rep.rows += len(df)
        firsts.append(df["date"].iloc[0])
        fresh += int(last >= latest)
    rep.coverage = rep.with_data / rep.universe if rep.universe else 0.0
    rep.fresh_frac = fresh / rep.with_data if rep.with_data else 0.0
    rep.earliest = min(firsts) if firsts else None
    years = Counter(d.year for d in firsts)
    rep.earliest_by_year = {y: years[y] for y in sorted(years)}
    if paths.issues.exists():
        iss = pd.read_parquet(paths.issues)
        eq_iss = iss[iss["symbol"].isin(set(uni["symbol"]))]
        rep.rejected_rows = int((eq_iss["severity"] == "reject").sum())
        rep.quarantined_rows = int(
            eq_iss.loc[eq_iss["severity"] == "quarantine", ["symbol", "date"]]
            .drop_duplicates()
            .shape[0]
        )
        rep.issue_counts = {str(k): int(v) for k, v in iss["check"].value_counts().items()}
    if paths.adjustments.exists():
        rep.adjustments = len(pd.read_parquet(paths.adjustments))
    rep.reject_frac = rep.rejected_rows / rep.rows if rep.rows else 0.0
    cal = TradingCalendar.from_file(cfg.settings.market, paths.calendar)
    rep.sessions = len(cal.sessions)
    rep.calendar_first = cal.first
    rep.probes = [
        probe_known_event(eq.read(ev.symbol), ev, vc.known_event_tolerance)
        for ev in vc.known_events
    ]
    rep.checks = {
        f"universe >= {vc.min_universe_symbols}": rep.universe >= vc.min_universe_symbols,
        f"coverage >= {vc.min_coverage:.0%}": rep.coverage >= vc.min_coverage,
        f"rejected rows <= {vc.max_reject_row_frac:.1%}": rep.reject_frac <= vc.max_reject_row_frac,
        f"fresh (last session) >= {vc.min_fresh_frac:.0%}": rep.fresh_frac >= vc.min_fresh_frac,
        "calendar built": rep.sessions > 0,
    }
    return rep
