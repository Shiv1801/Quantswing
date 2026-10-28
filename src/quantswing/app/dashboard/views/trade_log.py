"""D40 Trade Log page: live open positions (updated on every 5-min poll) and the full
closed-trade history. Paper only."""

from __future__ import annotations

import streamlit as st

from quantswing.app.dashboard.ui import Ctx, muted
from quantswing.app.dashboard.views.live import holdings
from quantswing.live.book import LiveBook
from quantswing.live.poller import short


def render(c: Ctx) -> None:
    """Render the page."""
    st.title("Trade Log")
    book = LiveBook(c.paths.live_db)

    # ------------------------------------------------------------ open positions (live)
    st.subheader("Open positions")
    h = holdings(book)
    if h.empty:
        st.caption("No open positions.")
    else:
        qt = book.quotes()
        if not qt.empty:
            muted(f"Live prices as of {qt['ts'].max()} · refreshes every poll")
        st.dataframe(
            h,
            hide_index=True,
            width="stretch",
            column_config={
                "Buy ₹": st.column_config.NumberColumn(format="%.2f"),
                "Live ₹": st.column_config.NumberColumn(format="%.2f"),
                "P&L ₹": st.column_config.NumberColumn(format="%+.0f"),
                "P&L %": st.column_config.NumberColumn(format="percent"),
                "SL-M trigger ₹": st.column_config.NumberColumn(format="%.2f"),
                "To stop %": st.column_config.NumberColumn(format="percent"),
                "Target ₹": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    # ------------------------------------------------------------ poll-by-poll history
    st.subheader("Price & P&L history (every poll)")
    qlog = book.quote_log(limit=3000)
    if qlog.empty:
        st.caption("No live polls recorded yet — this fills in once the 5-min poll runs.")
    else:
        held_syms = sorted(qlog.loc[qlog["qty"].notna(), "symbol"].unique())
        syms = ["All held positions"] + [short(s) for s in held_syms]
        pick = st.selectbox("Stock", syms, index=0)
        q = qlog if pick == "All held positions" else qlog[qlog["symbol"].map(short) == pick]
        q = q[q["qty"].notna()] if pick == "All held positions" else q
        log = q.assign(Stock=q["symbol"].map(short))[
            ["ts", "Stock", "ltp", "qty", "entry_px", "stop", "target", "pnl", "pnl_pct"]
        ].rename(
            columns={
                "ts": "Polled at",
                "ltp": "Live ₹",
                "qty": "Qty",
                "entry_px": "Buy ₹",
                "stop": "SL-M ₹",
                "target": "Target ₹",
                "pnl": "P&L ₹",
                "pnl_pct": "P&L %",
            }
        )
        st.dataframe(
            log,
            hide_index=True,
            width="stretch",
            height=360,
            column_config={
                "Live ₹": st.column_config.NumberColumn(format="%.2f"),
                "Buy ₹": st.column_config.NumberColumn(format="%.2f"),
                "SL-M ₹": st.column_config.NumberColumn(format="%.2f"),
                "Target ₹": st.column_config.NumberColumn(format="%.2f"),
                "P&L ₹": st.column_config.NumberColumn(format="%+.0f"),
                "P&L %": st.column_config.NumberColumn(format="percent"),
            },
        )
        muted("One row per 5-min poll while the stock was held. Newest first.")

    # ------------------------------------------------------------ closed trades
    st.subheader("Closed trades")
    tr = book.trades()
    if tr.empty:
        st.caption("No closed trades yet.")
    else:
        t = tr.assign(Stock=tr["symbol"].map(short))[
            [
                "Stock",
                "entry_date",
                "entry_px",
                "qty",
                "exit_date",
                "exit_px",
                "exit_reason",
                "sessions",
                "pnl",
                "return_pct",
                "r_multiple",
            ]
        ]
        st.dataframe(
            t.rename(
                columns={
                    "entry_date": "Bought",
                    "entry_px": "Buy ₹",
                    "qty": "Qty",
                    "exit_date": "Sold",
                    "exit_px": "Sell ₹",
                    "exit_reason": "Why",
                    "sessions": "Days",
                    "pnl": "P&L ₹",
                    "return_pct": "Return",
                    "r_multiple": "R",
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Buy ₹": st.column_config.NumberColumn(format="%.2f"),
                "Sell ₹": st.column_config.NumberColumn(format="%.2f"),
                "P&L ₹": st.column_config.NumberColumn(format="%+.0f"),
                "Return": st.column_config.NumberColumn(format="percent"),
                "R": st.column_config.NumberColumn(format="%.2f"),
            },
        )
