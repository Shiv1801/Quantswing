"""D40 end-of-day run: research refresh → replay the new session(s) through the D37 trade
engine (exactly the backtest code, daily bars) → the action list for the next session.

Run from the app ("Run end-of-day update") or ``python scripts/live_eod.py``. Safe to run
twice: an up-to-date book is left unchanged; missed days are caught up exactly.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from quantswing.alpha.research import AUX, composite_variant, edge_ic_series
from quantswing.core.calendar import TradingCalendar
from quantswing.core.config import AppConfig
from quantswing.live.book import LiveBook
from quantswing.portfolio.swing import SwingSim, SwingState
from quantswing.store.panels import FactorStore, load_panels
from quantswing.store.repositories import DataPaths

H10 = "COMPOSITE_H10"
EXIT_TEXT = {
    "Score": "Model exit: the score fell below average",
    "Time": "Time exit: held {held} sessions (limit {limit})",
    "Replaced": "Swap: replaced by a clearly stronger stock",
}
STOP_TEXT = {
    "SL": "Stop-loss",
    "BE": "Break-even stop (half already sold at target)",
    "TSL": "Trailing stop",
    "MacroTSL": "Trailing stop (tightened)",
}


@dataclass
class EodReport:
    """What one end-of-day run did."""

    status: str = "ok"
    sessions: list[str] = field(default_factory=list)
    next_session: str = ""
    equity: float = float("nan")
    day_pnl: float = float("nan")
    actions: list[dict[str, Any]] = field(default_factory=list)
    opened: bool = False
    message: str = ""


def research(
    paths: DataPaths, cfg: AppConfig, say: Callable[[str], None]
) -> tuple[pd.DataFrame, pd.Series]:
    """Factors → the D37 10-day score (significant factors, no macro factors)."""
    import time

    from quantswing.alpha.research import build_factors

    t0 = time.monotonic()
    n = [0]

    def tick(name: str) -> None:
        n[0] += 1
        say(f"  {n[0]:>2}. {name:<14} ({time.monotonic() - t0:.0f}s)")

    say("Loading prices and recomputing every factor (about 2-6 min; each line = one factor)...")
    skip = () if cfg.risk.swing.use_macro_factors else ("macro",)  # D37 does not use them
    build_factors(paths, cfg, progress=tick, skip=skip)
    store = FactorStore(paths.factors)
    mac = (
        set()
        if cfg.risk.swing.use_macro_factors
        else {n for n in store.names("neutral") if n.startswith("MAC_")}
    )
    say(
        f"Factors done in {time.monotonic() - t0:.0f}s. Building the 10-day score (each line = one factor's accuracy)..."
    )
    n[0] = 0
    t0 = time.monotonic()
    return composite_variant(paths, cfg, 10, 2.0, H10, progress=tick, exclude=mac)


def _money(x: float) -> str:
    return f"₹{x:,.2f}" if x < 1000 else f"₹{x:,.0f}"


def plan_actions(
    sim: SwingSim,
    i: int,
    prev: SwingState | None,
    new: SwingState,
    next_session: date,
) -> list[dict[str, Any]]:
    """Human-readable action list for ``next_session`` from the engine state at close ``i``."""
    sw = sim.sw
    nxt = str(next_session)
    out: list[dict[str, Any]] = []
    prev_stop = {
        int(p["trade_id"]): (float(p["stop"]), str(p["stop_kind"]))
        for p in (prev.positions if prev else [])
    }
    sells_value = 0.0
    for p in new.positions:
        sym = str(p["symbol"])
        j = sim.col_of[sym]
        close = float(sim.cl[i, j]) if np.isfinite(sim.cl[i, j]) else float(p["entry_px"])
        if p.get("exit_flag"):
            why = EXIT_TEXT.get(p["exit_flag"], p["exit_flag"]).format(
                held=p["held"], limit=sw.max_hold
            )
            out.append(
                {
                    "session": nxt,
                    "at": "09:15 at the open",
                    "symbol": sym,
                    "action": "SELL",
                    "qty": int(p["qty"]),
                    "order_type": "Market at open (cancel its SL-M / target orders first)",
                    "price": close,
                    "reason": why,
                    "key": f"{nxt}|{sym}|SELL|{p['trade_id']}",
                    "detail": {"trade_id": p["trade_id"], "flag": p["exit_flag"]},
                }
            )
            sells_value += int(p["qty"]) * close
            continue
        old = prev_stop.get(int(p["trade_id"]))
        stop, kind = float(p["stop"]), str(p["stop_kind"])
        if old is not None and stop > old[0] + 1e-9:
            out.append(
                {
                    "session": nxt,
                    "at": "Before 09:15",
                    "symbol": sym,
                    "action": "MODIFY SL-M",
                    "qty": int(p["qty"]),
                    "order_type": f"Stop-loss market: trigger {_money(old[0])} → {_money(stop)}",
                    "price": stop,
                    "reason": f"{STOP_TEXT.get(kind, kind)} raised: highest close "
                    f"{_money(float(p['peak']))} − {sw.tsl_sigma:g}σ₁₀"
                    if kind in ("TSL", "MacroTSL")
                    else f"{STOP_TEXT.get(kind, kind)}: stop moved to the buy price",
                    "key": f"{nxt}|{sym}|MODSL|{p['trade_id']}|{stop:.2f}",
                    "detail": {"trade_id": p["trade_id"], "old": old[0], "new": stop},
                }
            )
    # new buys for the next open
    elig = sim.elig[i] & np.isfinite(sim.z[i])
    zs = np.sort(sim.z[i][elig])[::-1]
    cash_est = new.cash + sells_value
    for b in new.buys:
        sym = str(b["symbol"])
        j = sim.col_of[sym]
        z, rf, eq = float(b["z"]), float(b["risk"]), float(b["equity"])
        px = float(sim.cl[i, j])
        s10, sd = float(sim.s10[i, j]), float(sim.sig[i, j])
        cap = px * (1 + sw.gap_skip_sigma * sd)
        sl = px * (1 - sw.sl_sigma * s10)
        q = math.floor(rf * eq / max(px - sl, 1e-9))
        q = max(0, min(q, math.floor(sw.max_weight * eq / px), math.floor(cash_est / (px * 1.003))))
        cash_est -= q * px
        rank = int((zs > z).sum()) + 1
        edge = float(sim.edge(i, j))
        out.append(
            {
                "session": nxt,
                "at": "09:15 at the open",
                "symbol": sym,
                "action": "BUY",
                "qty": q,
                "order_type": f"Limit {_money(cap)} at the open (skip if it opens higher)",
                "price": px,
                "reason": (
                    f"New buy: score z {z:.2f} (#{rank} of {int(elig.sum())}); expected "
                    f"+{edge:.1%} in 10 days; risk {_money(rf * eq)} ({rf:.2%} of capital). "
                    f"≈ qty at ₹{px:,.2f}; final qty is set at the open price"
                ),
                "key": f"{nxt}|{sym}|BUY",
                "detail": {
                    "z": z,
                    "rf": rf,
                    "equity": eq,
                    "s10": s10,
                    "sigma": sd,
                    "cap": cap,
                    "prev_close": px,
                    "sl_mult": sw.sl_sigma,
                    "tgt_mult": sw.target_sigma,
                    "tgt_frac": sw.target_sell_frac,
                    "max_weight": sw.max_weight,
                },
            }
        )
    return out


def record_session(
    sim: SwingSim, fills: pd.DataFrame, book: LiveBook, trades: pd.DataFrame, new: SwingState
) -> None:
    """Close out the planned actions of the replayed session(s) with the official fills."""
    if fills.empty:
        return
    lv: dict[int, tuple[float, float]] = {
        int(p["trade_id"]): (float(p["sl0"]), float(p["target"])) for p in new.positions
    }
    if not trades.empty:
        for t in trades.to_dict("records"):
            lv.setdefault(int(t["trade_id"]), (float(t["sl"]), float(t["target"])))
    for r in fills.to_dict("records"):
        d = str(pd.Timestamp(r["date"]).date())
        sym, why, q, px = str(r["symbol"]), str(r["reason"]), int(r["qty"]), float(r["px"])
        tid = int(r["trade_id"])
        if r["side"] == "buy":
            book.set_action(f"{d}|{sym}|BUY", f"Filled (paper) @ {_money(px)}", qty=q)
            tgt_q = q - math.floor(q * (1 - sim.sw.target_sell_frac))
            sl0, tgt = lv.get(tid, (float("nan"), float("nan")))
            book.add_action(
                {
                    "session": d,
                    "at": "Right after the buy",
                    "symbol": sym,
                    "action": "PLACE SL-M",
                    "qty": q,
                    "order_type": f"Stop-loss market, trigger {_money(sl0)} (resting)",
                    "price": sl0,
                    "reason": f"Stop-loss = buy − {sim.sw.sl_sigma:g}σ₁₀ ({sl0 / px - 1:+.1%})",
                    "status": "Done (paper)",
                    "key": f"{d}|{sym}|PLACE_SL|{tid}",
                }
            )
            book.add_action(
                {
                    "session": d,
                    "at": "Right after the buy",
                    "symbol": sym,
                    "action": "PLACE TARGET",
                    "qty": tgt_q,
                    "order_type": f"Limit sell at {_money(tgt)} (resting)",
                    "price": tgt,
                    "reason": f"Target = buy + {sim.sw.target_sigma:g}σ₁₀ ({tgt / px - 1:+.1%}):"
                    " sell half, then the SL moves to the buy price",
                    "status": "Done (paper)",
                    "key": f"{d}|{sym}|PLACE_TGT|{tid}",
                }
            )
            continue
        if why in ("Score", "Time", "Replaced"):
            book.set_action(f"{d}|{sym}|SELL|{tid}", f"Filled (paper) @ {_money(px)}")
            continue
        book.add_action(  # intraday stop / target / delisting — recorded as done
            {
                "session": d,
                "at": "During the session",
                "symbol": sym,
                "action": "SELL" if why != "Target" else "SELL (half)",
                "qty": q,
                "order_type": "Target limit" if why == "Target" else "SL-M triggered",
                "price": px,
                "reason": STOP_TEXT.get(why, why) + (" hit" if why != "Delisted" else ""),
                "status": f"Filled (paper) @ {_money(px)}",
                "key": f"{d}|{sym}|{why}|{tid}",
            }
        )


def eod_step(
    paths: DataPaths,
    cfg: AppConfig,
    book: LiveBook,
    comp: pd.DataFrame,
    ic: pd.Series,
    say: Callable[[str], None] = lambda _m: None,
) -> EodReport:
    """Replay every new session and plan the next one."""
    rep = EodReport()
    say("Loading prices...")
    p = load_panels(paths, cfg)
    store = FactorStore(paths.factors)
    gate = store.read(AUX, "gate").reindex_like(p.close).fillna(0).astype(bool)
    sim = SwingSim(p, comp.reindex_like(p.close), gate, cfg, edge_ic=edge_ic_series(ic, 10))
    idx = sim.idx
    i1 = len(idx) - 1
    prev = book.state()
    if prev is None:  # open the account at this close: first buys at the next open
        state = sim.init_state(book.capital(cfg.paper.live.capital))
        i0 = i1
        rep.opened = True
    else:
        state = prev
        i0 = int(idx.searchsorted(pd.Timestamp(str(state.last_session)), side="right"))
    if i0 > i1:
        rep.status = "up to date"
        rep.message = f"Already up to date (last session {state.last_session})."
        return rep
    say(f"Replaying {i1 - i0 + 1} session(s) through the trade engine...")
    new, res = sim.step(state, i0, i1, plan_last=True)
    cal = TradingCalendar.from_config()
    nxt = cal.next_trading_day(idx[i1].date())
    actions = plan_actions(sim, i1, prev, new, nxt)
    fills = res.fills
    rep.sessions = [str(d.date()) for d in idx[i0 : i1 + 1]]
    book.save_eod(new, res.daily, fills, res.trades, actions, p.bench)
    book.close_sessions(rep.sessions)
    record_session(sim, fills, book, res.trades, new)
    rep.next_session = str(nxt)
    rep.actions = actions
    eq = book.equity()
    rep.equity = float(new.equity)
    rep.day_pnl = float(eq["equity"].diff().iloc[-1]) if len(eq) > 1 else 0.0
    return rep


def eod_text(rep: EodReport, book: LiveBook) -> str:
    """Telegram / log summary."""
    if rep.status != "ok":
        return f"QuantSwing EOD: {rep.message}"
    lines = [
        f"QuantSwing EOD {', '.join(rep.sessions[-1:])}"
        + (" — account opened" if rep.opened else ""),
        f"Equity {_money(rep.equity)}  (day {rep.day_pnl:+,.0f})",
        f"Actions for {rep.next_session}: {len(rep.actions)}",
    ]
    for a in rep.actions[:15]:
        lines.append(
            f"- {a['action']} {a['symbol'].replace('NSE:', '').replace('-EQ', '')} "
            f"x{a['qty']}: {a['reason'][:80]}"
        )
    return "\n".join(lines)
