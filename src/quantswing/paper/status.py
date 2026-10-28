"""P6 exit test: 2 weeks unattended paper run with no crashes (07), plus integrity checks.

* sessions processed ≥ ``exit_test.min_sessions``;
* no failed daily runs in that window and no unresolved incidents;
* reconciliation (08 §4): positions = replay of filled orders;
* paper vs backtest replay (G3 groundwork): re-running the backtest from the paper start
  on today's data gives the same equity within ``max_replay_gap``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantswing.alpha.events import event_block
from quantswing.alpha.research import AUX, COMPOSITE
from quantswing.core.config import AppConfig
from quantswing.execution.paper_broker import PaperBook
from quantswing.paper.engine import reconcile
from quantswing.portfolio.construction import run_portfolio
from quantswing.store.panels import FactorStore, load_panels
from quantswing.store.repositories import DataPaths


@dataclass
class PaperStatus:
    """Numbers + verdicts for ``paper_status.py``."""

    start: str | None = None
    sessions: int = 0
    equity: float = float("nan")
    ret: float = float("nan")
    max_dd: float = float("nan")
    failed_runs: int = 0
    open_incidents: int = 0
    replay_gap: float = float("nan")
    mismatches: dict[str, float] = field(default_factory=dict)
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """All checks true."""
        return bool(self.checks) and all(self.checks.values())


def replay_gap(paths: DataPaths, cfg: AppConfig, eq: pd.Series, start: str) -> float:
    """Max relative gap between paper equity and a backtest replay from ``start``."""
    from quantswing.portfolio.p5 import load_regime

    p = load_panels(paths, cfg)
    store = FactorStore(paths.factors)
    gate = store.read(AUX, "gate").reindex_like(p.close).fillna(0).astype(bool)
    scores = store.read("composite", COMPOSITE).reindex_like(p.close)
    reg = load_regime(paths)
    res = run_portfolio(
        p,
        scores,
        gate,
        cfg,
        regime_mult=reg["mult"] if reg is not None else None,
        start=pd.Timestamp(start),
        entry_block=event_block(p, cfg.risk.event_window),
    )
    bt = res.daily["equity"].reindex(eq.index)
    ok = bt.notna() & eq.notna()
    if not ok.any():
        return float("nan")
    return float((eq[ok] / bt[ok] - 1).abs().max())


def paper_status(
    paths: DataPaths, cfg: AppConfig, book: PaperBook, replay: bool = True
) -> PaperStatus:
    """Compute the P6 exit test."""
    st = PaperStatus(start=book.get("start_date"))
    eq = book.equity()
    xt = cfg.paper.exit_test
    if eq.empty or st.start is None:
        st.checks = {"paper account started": False}
        return st
    st.sessions = len(eq)
    st.equity = float(eq["equity"].iloc[-1])
    st.ret = st.equity / float(book.get("start_capital", eq["equity"].iloc[0])) - 1
    st.max_dd = float(eq["dd"].max())
    window_start = eq.index[max(len(eq) - xt.min_sessions, 0)]
    runs = book.runs()
    if not runs.empty:
        started = pd.to_datetime(runs["started"])
        st.failed_runs = int(((runs["status"] == "failed") & (started >= window_start)).sum())
    st.open_incidents = len(book.incidents(open_only=True))
    st.mismatches = reconcile(book)
    if replay:
        st.replay_gap = replay_gap(paths, cfg, eq["equity"], st.start)
    st.checks = {
        f"sessions >= {xt.min_sessions}": st.sessions >= xt.min_sessions,
        "no failed runs in the window": st.failed_runs == 0,
        "no unresolved incidents": st.open_incidents == 0,
        "positions = replay of fills": not st.mismatches,
        f"paper vs backtest replay within {xt.max_replay_gap:.1%}": (
            bool(np.isfinite(st.replay_gap) and st.replay_gap <= xt.max_replay_gap)
            if replay
            else True
        ),
    }
    return st
