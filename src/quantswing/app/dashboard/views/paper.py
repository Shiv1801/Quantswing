"""Trading screens from the P6 paper book (06 §4 #3–#5): Watchlist (next-open plan),
Positions, Orders & Journal. Read-only; controls are via Telegram (/pause, /flatten yes)."""

from __future__ import annotations

import altair as alt
import streamlit as st

from quantswing.app.dashboard.ui import Ctx, muted
from quantswing.execution.order_state import OPEN_STATES
from quantswing.execution.paper_broker import PaperBook


def _book(c: Ctx) -> PaperBook | None:
    if not c.paths.paper_db.exists():
        st.info("Paper trading has not started yet. Run: python scripts\\paper_daily.py")
        return None
    return PaperBook(c.paths.paper_db)


def _name(sym: str) -> str:
    return sym.split(":")[-1].rsplit("-", 1)[0]


def watchlist(c: Ctx) -> None:
    """Orders planned for the next open."""
    st.title("Watchlist")
    book = _book(c)
    if book is None:
        return
    o = book.open_orders()
    if o.empty:
        st.success("No orders planned for the next open.")
        return
    o["stock"] = o["symbol"].map(_name)
    st.caption(f"Planned at the close of {o['decided_on'].max()} - executed at the next open.")
    st.dataframe(
        o[["stock", "side", "target_weight", "reason", "state", "model_version"]],
        width="stretch",
        hide_index=True,
        column_config={"target_weight": st.column_config.NumberColumn(format="percent")},
    )
    if book.control("pause_entries"):
        st.warning("New entries are paused (/resume on Telegram).")


def positions(c: Ctx) -> None:
    """Holdings + equity curve."""
    st.title("Positions")
    book = _book(c)
    if book is None:
        return
    eq = book.equity()
    if not eq.empty:
        last = eq.iloc[-1]
        start = float(book.get("start_capital", eq["equity"].iloc[0]))
        k = st.columns(5)
        k[0].metric("Equity", f"Rs {last['equity']:,.0f}", f"{last['equity'] / start - 1:+.2%}")
        k[1].metric("Invested", f"{last['invested'] / last['equity']:.0%}")
        k[2].metric("Positions", int(last["positions"]))
        k[3].metric("Drawdown", f"{last['dd']:.1%}")
        k[4].metric("Regime", str(last["regime"] or "-"))
        d = eq.reset_index()[["date", "equity"]]
        st.altair_chart(
            alt.Chart(d)
            .mark_line()
            .encode(x="date:T", y=alt.Y("equity:Q", scale=alt.Scale(zero=False))),
            width="stretch",
        )
    pos = book.positions()
    if pos.empty:
        muted("No holdings.")
        return
    pos["stock"] = pos["symbol"].map(_name)
    st.dataframe(
        pos[["stock", "qty", "last_px", "value", "weight", "since"]],
        width="stretch",
        hide_index=True,
        column_config={"weight": st.column_config.NumberColumn(format="percent")},
    )


def orders(c: Ctx) -> None:
    """Order book + state-machine journal."""
    st.title("Orders & Journal")
    book = _book(c)
    if book is None:
        return
    o = book.orders()
    if o.empty:
        muted("No orders yet.")
        return
    o["stock"] = o["symbol"].map(_name)
    done = o[~o["state"].isin(list(OPEN_STATES))]
    k = st.columns(4)
    fills = done[done["state"].isin(["FILLED", "CLOSED"])]
    k[0].metric("Fills", len(fills))
    k[1].metric("Traded value", f"Rs {fills['value'].sum():,.0f}")
    k[2].metric("Charges", f"Rs {fills['cost'].sum():,.0f}")
    k[3].metric("Expired / blocked", int(done["state"].isin(["EXPIRED"]).sum()))
    st.dataframe(
        o.sort_values("id", ascending=False)[
            [
                "id",
                "decided_on",
                "exec_session",
                "stock",
                "side",
                "qty",
                "fill_px",
                "cost",
                "state",
                "reason",
            ]
        ],
        width="stretch",
        hide_index=True,
        height=360,
    )
    st.subheader("State log")
    lg = book.order_log()
    st.dataframe(
        lg.sort_values("id", ascending=False).head(500),
        width="stretch",
        hide_index=True,
        height=300,
    )
    runs = book.runs()
    if not runs.empty:
        st.subheader("Daily runs")
        st.dataframe(
            runs.sort_values("id", ascending=False).head(30), width="stretch", hide_index=True
        )
    inc = book.incidents(open_only=True)
    if not inc.empty:
        st.error(f"{len(inc)} unresolved incident(s)")
        st.dataframe(inc, width="stretch", hide_index=True)
