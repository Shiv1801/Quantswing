"""Paper engine (P6, D32): step the D29 portfolio one session at a time on live data.

Run once per trading day after the EOD data update (``scripts/paper_daily.py``):

1. research refresh — factors → composite (IC-IR, point-in-time) → regime;
2. every session since the last run (catch-up is exact: decisions for day *t* only use data
   ≤ *t*): fill yesterday's plan at today's open, credit dividends, mark at the close;
3. decide the plan for the next open (same :class:`PortfolioSim` as the backtest);
4. persist orders / state log / equity / positions / simulator state in SQLite.

Operator controls (04 §8): ``pause_entries`` (no new names), ``flatten`` (sell everything
at the next open; one-shot). The first run opens the account (all cash) and plans the
first buys for the next open.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantswing.alpha.events import event_block
from quantswing.alpha.research import AUX, COMPOSITE
from quantswing.core.config import AppConfig, config_dir
from quantswing.execution.order_state import OrderState
from quantswing.execution.paper_broker import PaperBook
from quantswing.portfolio.construction import Controls, PortfolioSim, SimState
from quantswing.store.panels import FactorStore, load_panels
from quantswing.store.repositories import DataPaths

STRATEGY = "portfolio_regime"


def model_version(cfg: AppConfig) -> str:
    """``model-vX.Y.Z+<8-char hash of every config file>`` (08 §5)."""
    h = hashlib.sha1()
    for f in sorted(config_dir().glob("*.yaml")):
        h.update(f.read_bytes())
    return f"{cfg.settings.app.model_version}+{h.hexdigest()[:8]}"


@dataclass
class StepReport:
    """What one engine run did."""

    sessions: list[str] = field(default_factory=list)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    plan: pd.DataFrame = field(default_factory=pd.DataFrame)  # symbol, side, target_weight, reason
    daily: pd.DataFrame = field(default_factory=pd.DataFrame)
    regime: str | None = None
    regime_mult: float | None = None
    start_date: str | None = None
    opened: bool = False
    controls: dict[str, bool] = field(default_factory=dict)
    model_version: str = ""


def research_refresh(
    paths: DataPaths, cfg: AppConfig, progress: Callable[[str], None] | None = None
) -> None:
    """Factors → composite → regime (what the backtests used, rebuilt on today's data)."""
    from quantswing.alpha.research import build_factors, ic_analysis
    from quantswing.factors.gate import universe_gate
    from quantswing.portfolio.p5 import build_regime

    say = progress or (lambda _m: None)
    say("  factors")
    build_factors(paths, cfg)
    say("  composite (IC-IR)")
    ic_analysis(paths, cfg)
    say("  regime")
    p = load_panels(paths, cfg)
    build_regime(p, universe_gate(p, cfg.factors.gate), paths, cfg)


def _plan(
    pending: dict[str, float] | None, qty: dict[str, float], last: dict[str, float], equity: float
) -> pd.DataFrame:
    """Orders implied by next-open target weights vs current holdings."""
    rows: list[dict[str, Any]] = []
    for sym, w in (pending or {}).items():
        if not np.isfinite(w):
            continue
        q = qty.get(sym, 0.0)
        cw = q * last.get(sym, np.nan) / equity if q > 0 and equity > 0 else 0.0
        if q <= 0 and w > 0:
            rows.append({"symbol": sym, "side": "buy", "target_weight": w, "reason": "enter"})
        elif q > 0 and w <= 0:
            rows.append({"symbol": sym, "side": "sell", "target_weight": 0.0, "reason": "exit"})
        elif q > 0 and np.isfinite(cw) and abs(w - cw) > 1e-9:
            side = "buy" if w > cw else "sell"
            rows.append(
                {
                    "symbol": sym,
                    "side": side,
                    "target_weight": w,
                    "reason": "add" if w > cw else "trim",
                }
            )
    return pd.DataFrame(rows, columns=["symbol", "side", "target_weight", "reason"])


def step(paths: DataPaths, cfg: AppConfig, book: PaperBook) -> StepReport:
    """Process every new session and plan the next open. Idempotent when nothing is new."""
    from quantswing.portfolio.p5 import load_regime

    mv = model_version(cfg)
    rep = StepReport(model_version=mv)
    p = load_panels(paths, cfg)
    store = FactorStore(paths.factors)
    gate = store.read(AUX, "gate").reindex_like(p.close).fillna(0).astype(bool)
    scores = store.read("composite", COMPOSITE).reindex_like(p.close)
    reg = load_regime(paths)
    mult = reg["mult"] if reg is not None else None
    sim = PortfolioSim(
        p, scores, gate, cfg, regime_mult=mult, entry_block=event_block(p, cfg.risk.event_window)
    )
    idx = sim.idx
    i_to = len(idx) - 1
    saved = book.get("sim_state")
    if saved is None:
        state = sim.init_state(idx[i_to])
        i_from = i_to
        rep.opened = True
        book.put("start_date", str(idx[i_to].date()))
        book.put("start_capital", state.cash)
    else:
        state = SimState.from_json(saved)
        i_from = int(idx.searchsorted(pd.Timestamp(str(state.last_session)), side="right"))
    rep.start_date = book.get("start_date")
    ctl = Controls(pause_entries=book.control("pause_entries"), flatten=book.control("flatten"))
    rep.controls = {"pause_entries": ctl.pause_entries, "flatten": ctl.flatten}
    if reg is not None and len(reg["state"].dropna()):
        rep.regime = str(reg["state"].dropna().iloc[-1])
        rep.regime_mult = float(reg["mult"].dropna().iloc[-1])
    if i_from > i_to:
        return rep  # nothing new (weekend / holiday / already processed)
    new, daily, trades = sim.run(state, i_from, i_to, controls={idx[i_to]: ctl})
    rep.sessions = [str(idx[i].date()) for i in range(i_from, i_to + 1)]
    rep.trades = pd.DataFrame(trades)
    rep.daily = pd.DataFrame(daily)
    equity = float(daily[-1]["equity"]) if daily else state.cash
    rep.plan = _plan(new.pending, new.qty, new.last, equity)
    reg_state = reg["state"] if reg is not None else None
    with book.tx() as c:
        # fills for orders planned last time (+ system orders for catch-up / delisting)
        open_rows = c.execute(
            "SELECT id, symbol, side, state FROM orders WHERE state IN (?, ?, ?)",
            (OrderState.PENDING, OrderState.EXIT_PENDING, OrderState.EXIT_BLOCKED),
        ).fetchall()
        open_by = {(s, "buy" if sd == "buy" else "sell"): (oid, st) for oid, s, sd, st in open_rows}
        for t in trades:
            side = "buy" if t["side"] == "buy" else "sell"
            key = (str(t["symbol"]), side)
            sess = str(pd.Timestamp(t["date"]).date())
            if key in open_by:
                oid, st = open_by.pop(key)
                if st == OrderState.EXIT_BLOCKED:
                    book.move(c, oid, OrderState.EXIT_PENDING, reason="retry", mv=mv)
            else:
                prev = idx[max(int(idx.searchsorted(pd.Timestamp(t["date"]))) - 1, 0)]
                oid = book.place(
                    c, str(prev.date()), key[0], side, np.nan, str(t["reason"]), STRATEGY, mv
                )
            dst = OrderState.FILLED if side == "buy" else OrderState.CLOSED
            book.move(
                c,
                oid,
                dst,
                reason=str(t["reason"]),
                mv=mv,
                exec_session=sess,
                qty=float(t["qty"]),
                px=float(t["px"]),
                cost=float(t["cost"]),
            )
        carry = {k for k, v in (new.pending or {}).items() if np.isfinite(v)}
        for (sym, side), (oid, st) in open_by.items():
            if side == "sell" and sym in carry and st == OrderState.EXIT_PENDING:
                book.move(c, oid, OrderState.EXIT_BLOCKED, reason="lower circuit", mv=mv)
            elif st != OrderState.EXIT_BLOCKED or sym not in carry:
                book.move(c, oid, OrderState.EXPIRED, reason="not executed", mv=mv)
        # next-open plan (supersedes anything still open)
        for oid, _st in c.execute(
            "SELECT id, state FROM orders WHERE state IN (?, ?, ?)",
            (OrderState.PENDING, OrderState.EXIT_PENDING, OrderState.EXIT_BLOCKED),
        ).fetchall():
            book.move(c, oid, OrderState.CANCELLED, reason="superseded by new plan", mv=mv)
        for r in rep.plan.to_dict("records"):
            book.place(
                c,
                str(idx[i_to].date()),
                r["symbol"],
                r["side"],
                float(r["target_weight"]),
                r["reason"],
                STRATEGY,
                mv,
            )
        # equity + positions + state
        hist = list(state.eq_hist)
        for row in daily:
            hist.append(float(row["equity"]))
            peak = max(hist[-cfg.risk.dd_peak_window_sessions :])
            d = pd.Timestamp(row["date"])
            rs = None
            if reg_state is not None and d in reg_state.index and pd.notna(reg_state.loc[d]):
                rs = str(reg_state.loc[d])
            c.execute(
                "INSERT OR REPLACE INTO equity VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    str(d.date()),
                    row["equity"],
                    row["cash"],
                    row["invested"],
                    row["positions"],
                    int(row["halted"]),
                    row["gross_target"],
                    None if not np.isfinite(row["regime_mult"]) else row["regime_mult"],
                    rs,
                    1 - row["equity"] / peak if peak > 0 else 0.0,
                ),
            )
        c.execute("DELETE FROM positions")
        since = _since(c)
        for sym, q in new.qty.items():
            px = new.last.get(sym, np.nan)
            c.execute(
                "INSERT INTO positions VALUES (?,?,?,?,?,?)",
                (sym, q, px, q * px, q * px / equity if equity > 0 else 0.0, since.get(sym)),
            )
        book.put("sim_state", new.to_json(), c)
        book.put("last_session", new.last_session, c)
        book.put("model_version", mv, c)
        if ctl.flatten:
            c.execute("UPDATE controls SET value = '0' WHERE key = 'flatten'")
    return rep


def _since(c: Any) -> dict[str, str]:
    """Session each current holding was (re)opened, from replaying the fills."""
    rows = c.execute(
        "SELECT symbol, side, qty, exec_session FROM orders WHERE state IN ('FILLED','CLOSED') "
        "ORDER BY exec_session, id"
    ).fetchall()
    held: dict[str, float] = {}
    out: dict[str, str] = {}
    for sym, side, q, sess in rows:
        before = held.get(sym, 0.0)
        after = before + (q if side == "buy" else -q)
        if before <= 0 < after:
            out[sym] = sess
        if after <= 0:
            out.pop(sym, None)
        held[sym] = after
    return out


def reconcile(book: PaperBook) -> dict[str, float]:
    """08 §4: positions = replay of filled orders. Returns {symbol: qty mismatch} (empty = OK)."""
    o = book.orders()
    f = o[o["state"].isin(["FILLED", "CLOSED"])]
    signed = np.where(f["side"] == "buy", f["qty"], -f["qty"])
    replay = pd.Series(signed, index=f["symbol"]).groupby(level=0).sum()
    replay = replay[replay.abs() > 1e-9]
    pos = (
        book.positions().set_index("symbol")["qty"]
        if not book.positions().empty
        else pd.Series(dtype=float)
    )
    diff = replay.sub(pos, fill_value=0.0)
    return {str(k): float(v) for k, v in diff[diff.abs() > 1e-9].items()}
