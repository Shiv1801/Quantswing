"""EOD paper report (06 §3 "Reports"), plain text for Telegram and the log."""

from __future__ import annotations

import pandas as pd

from quantswing.execution.paper_broker import PaperBook
from quantswing.paper.engine import StepReport


def _name(sym: str) -> str:
    return sym.split(":")[-1].rsplit("-", 1)[0]


def eod_text(rep: StepReport, book: PaperBook) -> str:
    """Daily summary: equity, fills, next-open plan, controls."""
    eq = book.equity()
    if rep.opened:
        head = f"Paper account opened on {rep.start_date} with Rs {eq['equity'].iloc[-1]:,.0f}."
    elif not rep.sessions:
        return "No new trading session to process (holiday / weekend / already done)."
    else:
        head = f"Paper {', '.join(rep.sessions)}"
    lines = [head]
    if not eq.empty:
        last = eq.iloc[-1]
        prev = eq["equity"].iloc[-2] if len(eq) > 1 else last["equity"]
        start = float(book.get("start_capital", eq["equity"].iloc[0]))
        lines.append(
            f"Equity Rs {last['equity']:,.0f}  day {last['equity'] / prev - 1:+.2%}  "
            f"total {last['equity'] / start - 1:+.2%}  DD {last['dd']:.1%}"
        )
        lines.append(
            f"Positions {int(last['positions'])}  invested {last['invested'] / last['equity']:.0%}  "
            f"regime {rep.regime or '-'} (x{rep.regime_mult if rep.regime_mult is not None else 1:.1f})"
        )
    t = rep.trades
    if isinstance(t, pd.DataFrame) and not t.empty:
        lines.append("Fills:")
        for r in t.to_dict("records"):
            lines.append(
                f"  {str(r['side']).upper():<6} {_name(str(r['symbol'])):<12} "
                f"{float(r['qty']):>6.0f} @ {float(r['px']):,.2f}"
            )
    if not rep.plan.empty:
        lines.append("Plan for next open:")
        for r in rep.plan.to_dict("records"):
            lines.append(
                f"  {str(r['side']).upper():<4} {_name(str(r['symbol'])):<12} -> "
                f"{float(r['target_weight']):.1%} ({r['reason']})"
            )
    elif rep.sessions:
        lines.append("Plan for next open: no changes.")
    if rep.controls.get("pause_entries"):
        lines.append("Controls: new entries PAUSED (/resume to undo).")
    if rep.controls.get("flatten"):
        lines.append("Controls: FLATTEN applied - everything sells at the next open.")
    return "\n".join(lines)
