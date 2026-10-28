"""D40 live poll (every few minutes during market hours): prices for holdings and today's
planned buys → provisional paper events on screen and Telegram.

Same rules as the engine, applied to the live quote (open / day high / day low / last):

* planned BUY — once the stock has opened: skipped if it opened above the limit
  (prev close × (1 + 1σ)); otherwise "bought (paper)" at the open, quantity = risk ₹ ÷
  (open − SL) capped at 20% of capital and by cash; SL / target set from the open;
* planned SELL (score / time / swap exit) — "sold (paper)" at the open;
* holdings — day low ≤ stop → "SL / TSL hit"; day high ≥ target → "half sold at target,
  SL moved to the buy price".

The official book is written by the end-of-day run (daily bars, the backtest engine);
because the quote's day high / low are the same numbers, both agree on what happened.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any

import pandas as pd

from quantswing.alerts.telegram_bot import Telegram
from quantswing.core.calendar import TradingCalendar
from quantswing.core.clock import now_ist
from quantswing.core.config import AppConfig
from quantswing.live.book import LiveBook

OPEN, CLOSE = time(9, 15), time(15, 30)


@dataclass
class PollReport:
    """What one poll did."""

    status: str
    events: list[str] = field(default_factory=list)
    symbols: int = 0


def short(sym: str) -> str:
    """NSE:ABC-EQ → ABC."""
    return sym.replace("NSE:", "").rsplit("-", 1)[0]


def normalise_quotes(raw: pd.DataFrame) -> pd.DataFrame:
    """Fyers quote fields → symbol, ltp, open, high, low, prev_close."""
    if raw.empty:
        return pd.DataFrame(columns=["symbol", "ltp", "open", "high", "low", "prev_close"])

    def g(c: str) -> pd.Series:
        col = raw[c] if c in raw else pd.Series(float("nan"), index=raw.index)
        out: pd.Series = pd.to_numeric(col, errors="coerce")
        return out

    return pd.DataFrame(
        {
            "symbol": raw["symbol"].astype(str),
            "ltp": g("lp"),
            "open": g("open_price"),
            "high": g("high_price"),
            "low": g("low_price"),
            "prev_close": g("prev_close_price"),
        }
    )


def poll_once(
    book: LiveBook,
    cfg: AppConfig,
    quotes: Any,
    *,
    now: datetime | None = None,
    tg: Telegram | None = None,
    cal: TradingCalendar | None = None,
    force: bool = False,
) -> PollReport:
    """One live update. ``quotes(symbols) -> DataFrame`` (Fyers ``FyersData.quotes``)."""
    now = now or now_ist()
    cal = cal or TradingCalendar.from_config()
    today = now.date()
    if not force and (not cal.is_trading_day(today) or not (OPEN <= now.time() <= time(15, 40))):
        return PollReport("market closed")
    st = book.state()
    if st is None:
        return PollReport("account not open yet (run the end-of-day update once)")
    session = str(today)
    if st.last_session == session:
        return PollReport("today's end-of-day update already done")
    sw = cfg.risk.swing
    acts = book.actions(session)
    buys = acts[(acts["action"] == "BUY")] if not acts.empty else acts
    sells = (
        acts[(acts["action"] == "SELL") & acts["key"].str.contains(r"\|SELL\|")]
        if not acts.empty
        else acts
    )
    syms = sorted(
        {str(p["symbol"]) for p in st.positions} | set(buys["symbol"] if not buys.empty else [])
    )
    if not syms:
        return PollReport("nothing to watch today")
    q = normalise_quotes(quotes(syms))
    book.save_quotes(q)
    qd: dict[str, dict[str, Any]] = {
        str(r["symbol"]): {str(k): v for k, v in r.items()} for r in q.to_dict("records")
    }
    rep = PollReport("ok", symbols=len(syms))

    def emit(sym: str, kind: str, msg: str, key: str) -> None:
        if book.event(session, sym, kind, msg, f"{session}|{key}"):
            rep.events.append(msg)
            if tg is not None and tg.enabled:
                tg.send(f"QuantSwing (paper) {now:%H:%M}: {msg}")

    def opened(r: dict[str, Any] | None) -> bool:
        return r is not None and math.isfinite(r["open"]) and r["open"] > 0

    # --- exits flagged at yesterday's close: sold at the open
    cash = float(st.cash)
    for a in sells.to_dict("records") if not sells.empty else []:
        r = qd.get(a["symbol"])
        if opened(r) and str(a["status"]).startswith("Planned"):
            px = float(r["open"])  # type: ignore[index]
            cash += int(a["qty"]) * px
            book.set_action(a["key"], f"Sold (paper, live) @ ₹{px:,.2f}")
            emit(
                a["symbol"],
                "sell",
                f"SOLD {short(a['symbol'])} x{a['qty']} at the open ₹{px:,.2f} ({a['reason']})",
                f"{a['key']}",
            )
    exiting = {a["symbol"] for a in (sells.to_dict("records") if not sells.empty else [])}
    # --- planned buys: at the open
    live_pos: list[dict[str, Any]] = [dict(p) for p in st.positions if p["symbol"] not in exiting]
    for a in buys.to_dict("records") if not buys.empty else []:
        r = qd.get(a["symbol"])
        d = json.loads(a["detail"] or "{}")
        if not opened(r) or not d:
            continue
        o = float(r["open"])  # type: ignore[index]
        if str(a["status"]).startswith("Planned"):
            if o > float(d["cap"]):
                book.set_action(
                    a["key"], f"Skipped (live): opened ₹{o:,.2f} above the limit ₹{d['cap']:,.2f}"
                )
                emit(
                    a["symbol"],
                    "skip",
                    f"SKIPPED {short(a['symbol'])}: opened ₹{o:,.2f} above the limit ₹{d['cap']:,.2f}",
                    a["key"],
                )
                continue
            s10 = float(d["s10"])
            sl = o * (1 - float(d["sl_mult"]) * s10)
            tgt = o * (1 + float(d["tgt_mult"]) * s10)
            eq = float(d["equity"])
            qty = math.floor(float(d["rf"]) * eq / max(o - sl, 1e-9))
            qty = max(
                0,
                min(
                    qty, math.floor(float(d["max_weight"]) * eq / o), math.floor(cash / (o * 1.003))
                ),
            )
            if qty <= 0:
                book.set_action(a["key"], "Skipped (live): not enough cash")
                continue
            cash -= qty * o
            book.set_action(a["key"], f"Bought (paper, live) @ ₹{o:,.2f}", qty=qty)
            half = qty - math.floor(qty * (1 - float(d["tgt_frac"])))
            for act, order, px, why in (
                (
                    "PLACE SL-M",
                    f"Stop-loss market, trigger ₹{sl:,.2f} (resting)",
                    sl,
                    f"Stop-loss = buy − {d['sl_mult']:g}σ₁₀ ({sl / o - 1:+.1%})",
                ),
                (
                    "PLACE TARGET",
                    f"Limit sell at ₹{tgt:,.2f} (resting)",
                    tgt,
                    f"Target = buy + {d['tgt_mult']:g}σ₁₀ ({tgt / o - 1:+.1%}): sell half, then the SL moves to the buy price",
                ),
            ):
                book.add_action(
                    {
                        "session": session,
                        "at": "Right after the buy",
                        "symbol": a["symbol"],
                        "action": act,
                        "qty": qty if act == "PLACE SL-M" else half,
                        "order_type": order,
                        "price": px,
                        "reason": why,
                        "status": "Done (paper, live)",
                        "key": f"{session}|{a['symbol']}|{act.replace(' ', '_')}|live",
                    }
                )
            emit(
                a["symbol"],
                "buy",
                f"BOUGHT {short(a['symbol'])} x{qty} at the open ₹{o:,.2f}; SL ₹{sl:,.2f}, target ₹{tgt:,.2f}",
                a["key"],
            )
            live_pos.append(
                {
                    "symbol": a["symbol"],
                    "qty": qty,
                    "entry_px": o,
                    "stop": sl,
                    "stop_kind": "SL",
                    "target": tgt,
                    "partial": False,
                }
            )
        elif str(a["status"]).startswith("Bought"):
            s10 = float(d["s10"])
            live_pos.append(
                {
                    "symbol": a["symbol"],
                    "qty": int(a["qty"]),
                    "entry_px": o,
                    "stop": o * (1 - float(d["sl_mult"]) * s10),
                    "stop_kind": "SL",
                    "target": o * (1 + float(d["tgt_mult"]) * s10),
                    "partial": False,
                }
            )
    # --- holdings: stops and targets against the day's range
    for p in live_pos:
        sym = str(p["symbol"])
        r = qd.get(sym)
        if not opened(r):
            continue
        lo, hi, o = float(r["low"]), float(r["high"]), float(r["open"])  # type: ignore[index]
        stop, kind = float(p["stop"]), str(p.get("stop_kind", "SL"))
        tgt_hit_before = f"{session}|{sym}|TARGET" in _keys(book, session)
        if not p.get("partial") and not tgt_hit_before and hi >= float(p["target"]):
            half = int(p["qty"]) - math.floor(int(p["qty"]) * (1 - sw.target_sell_frac))
            emit(
                sym,
                "target",
                f"TARGET {short(sym)}: sold {half} at ₹{max(float(p['target']), o):,.2f}; SL moved to the buy price ₹{float(p['entry_px']):,.2f}",
                f"{sym}|TARGET",
            )
            stop, kind = max(stop, float(p["entry_px"])), "BE"
            tgt_hit_before = True
        if lo <= stop:
            px = min(o, stop)
            label = {"SL": "STOP-LOSS", "BE": "BREAK-EVEN STOP", "TSL": "TRAILING STOP"}.get(
                kind, kind
            )
            emit(
                sym,
                "stop",
                f"{label} {short(sym)}: sold at ~₹{px:,.2f} (day low ₹{lo:,.2f})",
                f"{sym}|STOP",
            )
    # persist this poll's view: today's live buys/sells folded in, so Holdings and the
    # Trade Log show something real intraday (the end-of-day run remains authoritative
    # and overwrites this with the official book)
    book.save_live_positions(session, live_pos)
    ltp_map = {sym: float(r["ltp"]) for sym, r in qd.items() if math.isfinite(r.get("ltp", math.nan))}
    book.log_positions(ltp_map, live_pos)
    return rep


def _keys(book: LiveBook, session: str) -> set[str]:
    ev = book.events(500)
    return set(ev.loc[ev["session"] == session, "key"]) if not ev.empty else set()
