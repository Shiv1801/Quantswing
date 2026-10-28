"""P2 research pipeline: factor panels → cross-sectional processing → IC report.

1. :func:`build_factors` — load adjusted panels, compute the gate and size control, then
   for each factor: raw → winsorise/z/neutralise on the gated universe → persist.
2. :func:`ic_analysis` — rank IC of every neutral factor vs forward returns at each
   horizon; IC-IR weights (point-in-time) and the composite's IC.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantswing.alpha import combine
from quantswing.alpha.ic import IcStats, forward_log_return, ic_by_year, rank_ic, summarise
from quantswing.core.config import AppConfig, get_config
from quantswing.core.logging import get_logger
from quantswing.factors import registry, xsec
from quantswing.factors.common import Panels
from quantswing.factors.gate import universe_gate
from quantswing.factors.liquidity import size_proxy
from quantswing.factors.nse_factors import sector_labels
from quantswing.store.panels import FactorStore, load_panels
from quantswing.store.repositories import DataPaths, write_parquet_frame

log = get_logger(__name__)
AUX = "aux"
COMPOSITE = "COMPOSITE"


@dataclass
class BuildSummary:
    """What :func:`build_factors` produced."""

    sessions: int = 0
    symbols: int = 0
    gated_avg: float = 0.0
    coverage: dict[str, float] = field(default_factory=dict)  # avg names/day with neutral z


def build_factors(
    paths: DataPaths,
    cfg: AppConfig | None = None,
    limit: int | None = None,
    progress: Callable[[str], None] | None = None,
    skip: tuple[str, ...] = (),
) -> BuildSummary:
    """Compute, process and store every factor (``skip``: families to leave as they are)."""
    cfg = cfg or get_config()
    fc = cfg.factors
    store = FactorStore(paths.factors)
    p = load_panels(paths, cfg, limit=limit)
    gate = universe_gate(p, fc.gate)
    size = size_proxy(p.turnover, fc.size_proxy.window)
    sectors = (
        sector_labels(p.sector, p.close.columns, fc.xsec.min_sector_names)
        if p.sector is not None and p.sector.notna().any()
        else None
    )
    store.write(AUX, "close", p.close)
    store.write(AUX, "gate", gate.astype("float32"))
    summ = BuildSummary(
        sessions=len(p.close), symbols=p.close.shape[1], gated_avg=float(gate.sum(axis=1).mean())
    )
    snaps: list[pd.DataFrame] = [_snapshot("GATE", gate.astype("float64"), gate.astype("float64"))]
    for name, raw in registry.compute_all(p, fc, skip):
        z, nz = xsec.process(raw, gate, size, fc.xsec, sectors)
        store.write("raw", name, raw)
        store.write("z", name, z)
        store.write("neutral", name, nz)
        snaps.append(_snapshot(name, raw, nz))
        summ.coverage[name] = float(nz.notna().sum(axis=1).mean())
        if progress:
            progress(name)
    write_parquet_frame(pd.concat(snaps, ignore_index=True), store.root / SNAPSHOT_FILE)
    return summ


def build_family(
    paths: DataPaths,
    cfg: AppConfig,
    family: str,
    p: Panels | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[str]:
    """Compute, process and store ONE factor family (e.g. D39 ``macro``) on top of an
    existing factor store (same gate, size proxy and sector labels). Returns factor names."""
    fc = cfg.factors
    store = FactorStore(paths.factors)
    p = p or load_panels(paths, cfg)
    gate = store.read(AUX, "gate").reindex_like(p.close).fillna(0).astype(bool)
    size = size_proxy(p.turnover, fc.size_proxy.window)
    sectors = (
        sector_labels(p.sector, p.close.columns, fc.xsec.min_sector_names)
        if p.sector is not None and p.sector.notna().any()
        else None
    )
    done: list[str] = []
    for name, raw in registry.compute_family(family, p, fc).items():
        z, nz = xsec.process(raw, gate, size, fc.xsec, sectors)
        store.write("raw", name, raw)
        store.write("z", name, z)
        store.write("neutral", name, nz)
        done.append(name)
        if progress:
            progress(name)
    return done


SNAPSHOT_FILE = "snapshot.parquet"
SNAPSHOT_SESSIONS = 5


def _snapshot(name: str, raw: pd.DataFrame, nz: pd.DataFrame) -> pd.DataFrame:
    """Long table of the last few sessions (date, symbol, factor, raw, neutral) for the UI."""
    tail_r, tail_n = raw.tail(SNAPSHOT_SESSIONS), nz.tail(SNAPSHOT_SESSIONS)
    dates = np.repeat(tail_r.index.to_numpy(), tail_r.shape[1])
    syms = np.tile(tail_r.columns.to_numpy(), tail_r.shape[0])
    out = pd.DataFrame(
        {
            "date": dates,
            "symbol": syms,
            "factor": name,
            "raw": tail_r.to_numpy(dtype="float64").ravel(),
            "neutral": tail_n.reindex_like(tail_r).to_numpy(dtype="float64").ravel(),
        }
    )
    res: pd.DataFrame = out.dropna(subset=["raw", "neutral"], how="all")
    return res


@dataclass
class IcReport:
    """IC analysis output."""

    stats: pd.DataFrame  # rows: factor × horizon
    by_year: pd.DataFrame  # factor × year at the primary horizon
    weights_last: pd.Series  # latest IC-IR weights
    horizons: list[int]
    primary: int


def ic_analysis(
    paths: DataPaths, cfg: AppConfig | None = None, progress: Callable[[str], None] | None = None
) -> IcReport:
    """Rank IC by horizon for every stored neutral factor + the IC-IR composite."""
    cfg = cfg or get_config()
    ic_cfg = cfg.factors.ic
    store = FactorStore(paths.factors)
    close = store.read(AUX, "close")
    logp = pd.DataFrame(np.log(close), index=close.index, columns=close.columns)
    fwd = {h: forward_log_return(logp, h) for h in ic_cfg.horizons}
    names = store.names("neutral")
    rows: list[dict[str, object]] = []
    primary_ic: dict[str, pd.Series] = {}
    years: dict[str, pd.Series] = {}

    def record(name: str, z: pd.DataFrame) -> None:
        for h in ic_cfg.horizons:
            ic = rank_ic(z, fwd[h], ic_cfg.min_names_per_date)
            st: IcStats = summarise(ic, h)
            rows.append({"factor": name, **st.__dict__})
            if h == ic_cfg.primary_horizon:
                primary_ic[name] = ic
                years[name] = ic_by_year(ic)

    for name in names:
        record(name, store.read("neutral", name))
        if progress:
            progress(name)

    ic_panel = pd.DataFrame(primary_ic)
    weights = combine.ic_ir_weights(ic_panel, ic_cfg.primary_horizon, ic_cfg.combine)
    comp = combine.composite(((n, store.read("neutral", n)) for n in names), weights)
    store.write("composite", COMPOSITE, comp)
    write_parquet_frame(_snapshot(COMPOSITE, comp, comp), store.root / "composite_snapshot.parquet")
    weights.to_parquet(paths.factors / "icir_weights.parquet")
    record(COMPOSITE, comp)
    last = weights.dropna(how="all")
    weights_last = last.iloc[-1] if len(last) else pd.Series(dtype="float64")
    return IcReport(
        stats=pd.DataFrame(rows),
        by_year=pd.DataFrame(years).T,
        weights_last=weights_last,
        horizons=list(ic_cfg.horizons),
        primary=ic_cfg.primary_horizon,
    )


def composite_variant(
    paths: DataPaths,
    cfg: AppConfig,
    horizon: int,
    min_t: float,
    name: str,
    progress: Callable[[str], None] | None = None,
    factor_ics: dict[str, pd.Series] | None = None,
    exclude: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Composite at another forecast ``horizon`` using only factors whose trailing IC
    t-stat ≥ ``min_t`` (both point-in-time). Stored as ``composite/<name>`` (the main
    composite is untouched). Returns (composite, its daily rank IC at ``horizon``)."""
    ic_cfg = cfg.factors.ic
    store = FactorStore(paths.factors)
    close = store.read(AUX, "close")
    logp = pd.DataFrame(np.log(close), index=close.index, columns=close.columns)
    fwd = forward_log_return(logp, horizon)
    names = [n for n in store.names("neutral") if n not in (exclude or set())]
    ics: dict[str, pd.Series] = {}
    for n in names:
        ics[n] = rank_ic(store.read("neutral", n), fwd, ic_cfg.min_names_per_date)
        if progress:
            progress(n)
    if factor_ics is not None:  # caller wants each factor's daily IC (diagnosis)
        factor_ics.update(ics)
    cc = ic_cfg.combine.model_copy(update={"min_t": min_t})
    weights = combine.ic_ir_weights(pd.DataFrame(ics), horizon, cc)
    comp = combine.composite(((n, store.read("neutral", n)) for n in names), weights)
    store.write("composite", name, comp)
    return comp, rank_ic(comp, fwd, ic_cfg.min_names_per_date)


def edge_ic_series(comp_ic: pd.Series, horizon: int, window: int = 252) -> pd.Series:
    """Point-in-time IC of the score: trailing mean of ICs whose window has closed."""
    known = comp_ic.shift(horizon)
    return known.rolling(window, min_periods=60).mean().clip(0.005, 0.2)


def ic_exposure_series(
    comp_ic: pd.Series, horizon: int, window: int = 63, full_t: float = 2.0
) -> pd.Series:
    """B: invested share from the score's own recent accuracy. t-stat of the trailing
    ``window`` daily ICs whose ``horizon`` has closed (overlap-adjusted: √(window/horizon)
    independent draws); share = clip(t / full_t, 0, 1); 1 until enough history."""
    known = comp_ic.shift(horizon)
    r = known.rolling(window, min_periods=max(window * 2 // 3, 10))
    m, s = r.mean(), r.std()
    t = m / (s / np.sqrt(window / horizon)).where(s > 0)
    out: pd.Series = (t / full_t).clip(0.0, 1.0).fillna(1.0)
    return out
