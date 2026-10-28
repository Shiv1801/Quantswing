"""D40 Actions page: what to do (and what the paper account did), live holdings, the
end-of-day button, capital. Paper only."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from quantswing.app.dashboard.ui import Ctx, muted
from quantswing.app.dashboard.views.login import login_chip
from quantswing.core.clock import now_ist
from quantswing.core.config import REPO_ROOT
from quantswing.live.book import LiveBook
from quantswing.live.poller import short
from quantswing.live.proc import kill_scripts, kill_tree

ACTION_ICON = {
    "BUY": "🟢 BUY",
    "SELL": "🔴 SELL",
    "SELL (half)": "🟠 SELL half",
    "MODIFY SL-M": "🟡 MODIFY SL-M",
    "PLACE SL-M": "🛡️ PLACE SL-M",
    "PLACE TARGET": "🎯 PLACE TARGET",
}


def _launch(c: Ctx, script: str, *args: str) -> None:
    """Run a script in the background (no console window); it writes its own log.
    Start-up errors (before the script's own log opens) go to logs/live_launch.log."""
    flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
    err = c.paths.live_eod_log.parent / "live_launch.log"
    err.parent.mkdir(parents=True, exist_ok=True)
    with err.open("a", encoding="utf-8") as f:
        f.write(f"\n--- {now_ist():%Y-%m-%d %H:%M:%S} {script} {' '.join(args)}\n")
        f.flush()
        subprocess.Popen(
            [sys.executable, str(REPO_ROOT / "scripts" / script), *args],
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=f,
            creationflags=flags,
        )


def _stop_eod(book: LiveBook) -> None:
    """Stop the end-of-day job (and any stray copies, e.g. from an older version)."""
    run = book.active_run("eod")
    if run is not None:
        kill_tree(int(run["pid"]))
        book.run_end(int(run["id"]), "stopped", "stopped by you")
    kill_scripts("live_eod.py")
    for row in book.runs("eod", 5).to_dict("records"):
        if row["status"] == "running":
            book.run_end(int(row["id"]), "stopped", "stopped by you")


def _tail(path: object, n: int = 25) -> str:
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()  # type: ignore[call-overload]  # noqa: SIM115
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def holdings(book: LiveBook) -> pd.DataFrame:
    """Open positions with the latest live price. Uses the official book (after the last
    end-of-day run); if today's live poll has since bought/sold anything the official book
    doesn't know about yet, those are shown too (marked "live", pending tonight's EOD)."""
    st_ = book.state()
    official = list(st_.positions) if st_ is not None else []
    live_only = book.live_positions()
    if not official and not live_only:
        return pd.DataFrame()
    qt = book.quotes()
    ltp_map = {
        str(k): float(v) for k, v in zip(qt.get("symbol", []), qt.get("ltp", []), strict=False)
    }
    rows = []
    for p, is_live in [(p, False) for p in official] + [(p, True) for p in live_only]:
        sym = str(p["symbol"])
        last = float(st_.last.get(sym, np.nan)) if st_ is not None else np.nan
        ltp = ltp_map.get(sym, np.nan)
        ltp = ltp if np.isfinite(ltp) else last
        qty, entry, stop = int(p["qty"]), float(p["entry_px"]), float(p["stop"])
        rows.append(
            {
                "Stock": short(sym),
                "Status": "live (pending EOD)" if is_live else "official",
                "Qty": qty,
                "Buy ₹": entry,
                "Live ₹": ltp,
                "P&L ₹": (ltp - entry) * qty,
                "P&L %": ltp / entry - 1,
                "SL-M trigger ₹": stop,
                "Stop type": p.get("stop_kind", "SL"),
                "To stop %": stop / ltp - 1 if ltp else np.nan,
                "Target ₹": np.nan if p.get("partial") else float(p.get("target", np.nan)),
                "Half sold": "yes" if p.get("partial") else "",
                "Days": int(p["held"]) if "held" in p else 0,
                "Next open": {
                    "Score": "SELL (score)",
                    "Time": "SELL (time)",
                    "Replaced": "SELL (swap)",
                }.get(str(p.get("exit_flag") or ""), "hold"),
            }
        )
    return pd.DataFrame(rows)


def render(c: Ctx) -> None:
    """Render the page."""
    st.title("Actions")
    login_chip()
    book = LiveBook(c.paths.live_db)
    st_ = book.state()
    eq = book.equity()

    # ------------------------------------------------------------ account
    if st_ is not None and not eq.empty:
        start = float(book.capital(c.cfg.paper.live.capital)) + float(
            book.cash_moves()["amount"].sum() if not book.cash_moves().empty else 0.0
        )
        last = eq.iloc[-1]
        n0 = eq["nifty"].dropna()
        nifty_ret = float(n0.iloc[-1] / n0.iloc[0] - 1) if len(n0) > 1 else np.nan
        k = st.columns(6)
        k[0].metric("Equity", f"₹{st_.equity:,.0f}", f"{st_.equity / start - 1:+.2%} since start")
        k[1].metric("Cash", f"₹{st_.cash:,.0f}")
        k[2].metric("Invested", f"{float(last['invested']) / float(last['equity']):.0%}")
        k[3].metric("Positions", len(st_.positions))
        k[4].metric("Nifty 500 since start", f"{nifty_ret:+.2%}" if np.isfinite(nifty_ret) else "—")
        k[5].metric("Last EOD", str(st_.last_session))
    else:
        st.info("The paper account opens at the first end-of-day update (below).", icon="🧾")

    # ------------------------------------------------------------ buttons
    b = st.columns([1.3, 1.3, 1.2, 3])
    eod_running = book.active_run("eod") is not None
    no_data = b[2].checkbox(
        "Data already updated", help="Tick, then click Run: skips the Fyers/NSE download"
    )
    if eod_running:
        if b[0].button("■ Stop end-of-day update", type="secondary"):
            _stop_eod(book)
            st.rerun()
    elif b[0].button("▶ Run end-of-day update", type="primary"):
        _stop_eod(book)  # clears stray jobs from older versions before starting
        _launch(c, "live_eod.py", *(["--no-data"] if no_data else []))
        time.sleep(2)
        st.rerun()
    if b[1].button("⟳ Update live prices now", disabled=eod_running):
        _launch(c, "live_poll.py")
        time.sleep(4)
        st.rerun()
    muted(
        "End of day: data → scores → replay the day → the action list for the next session "
        "(~5–15 min; run it after 18:30 so the NSE files are out). Live prices update "
        f"automatically every {c.cfg.paper.live.poll_minutes} min during market hours."
    )

    @st.fragment(run_every=3)
    def eod_progress() -> None:
        run = book.active_run("eod")
        was = st.session_state.get("eod_was_running", False)
        st.session_state["eod_was_running"] = run is not None
        if was and run is None:
            st.rerun()  # finished: refresh the whole page (buttons, actions, holdings)
        launch = _tail(c.paths.live_eod_log.parent / "live_launch.log", 200)
        launch = launch[launch.rfind("\n--- ") + 1 :] if "\n--- " in launch else launch
        if "Traceback" in launch or "Error" in launch:
            with st.expander("⚠️ Start-up messages (the last job may not have started)"):
                st.code(launch, language=None)
        if run is not None:
            st.info(f"End-of-day update running since {run['started']}...", icon="⏳")
            st.code(_tail(c.paths.live_eod_log), language=None)
        else:
            r = book.runs("eod", 1)
            if not r.empty:
                row = r.iloc[0]
                txt = f"Last end-of-day update: {row['status']} at {row['ended'] or row['started']}"
                (st.caption if row["status"] == "ok" else st.warning)(txt)
                if row["status"] != "ok":
                    with st.expander("Log", expanded=True):
                        st.code(_tail(c.paths.live_eod_log, 60), language=None)

    eod_progress()

    # ------------------------------------------------------------ actions
    acts = book.actions()
    st.subheader("Action list")
    if acts.empty:
        st.caption("No actions yet — run the end-of-day update.")
    else:
        sessions = sorted(acts["session"].unique(), reverse=True)
        sel = st.selectbox("Session", sessions, index=0, label_visibility="collapsed")
        a = acts[acts["session"] == sel].copy()
        a["Action"] = a["action"].map(lambda x: ACTION_ICON.get(x, x))
        a["Stock"] = a["symbol"].map(short)
        view = a.rename(
            columns={
                "session": "Date",
                "at": "When",
                "qty": "Qty",
                "order_type": "Order",
                "price": "Price ₹",
                "reason": "Reason",
                "status": "Status",
            }
        )[["Date", "When", "Stock", "Action", "Qty", "Price ₹", "Status", "Order", "Reason"]]
        view["Price ₹"] = view["Price ₹"].map(lambda x: "" if pd.isna(x) else f"{x:,.2f}")
        st.table(view.set_index("Stock"))  # full text wraps: every reason is readable
        muted(
            "Paper account only: statuses are filled in automatically (live updates during the day, official at the end-of-day update)."
        )

    # ------------------------------------------------------------ holdings
    st.subheader("Holdings (live)")
    h = holdings(book)
    if h.empty:
        st.caption("No open positions.")
    else:
        qt = book.quotes()
        if not qt.empty:
            muted(f"Live prices as of {qt['ts'].max()}")
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

    muted("For a poll-by-poll price/P&L history and every closed trade, see the Trade Log page.")

    ev = book.events(50)
    if not ev.empty:
        st.subheader("Live events")
        st.dataframe(
            ev[["ts", "message"]].rename(columns={"ts": "Time", "message": "Event"}),
            hide_index=True,
            width="stretch",
            height=220,
        )

    # ------------------------------------------------------------ equity + trades
    if len(eq) > 1:
        st.subheader("Paper account vs Nifty 500")
        e = eq.copy()
        e["date"] = pd.to_datetime(e["date"])
        base = float(e["equity"].iloc[0])
        e["Paper account"] = e["equity"]
        e["Nifty 500"] = e["nifty"] / e["nifty"].iloc[0] * base
        long = e.melt("date", ["Paper account", "Nifty 500"], "series", "₹")
        st.altair_chart(
            alt.Chart(long)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("₹:Q", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(range=["#2a78d6", "#eb6834"]),
                    legend=alt.Legend(orient="top", title=None),
                ),
                tooltip=["date:T", "series:N", alt.Tooltip("₹:Q", format=",.0f")],
            )
            .properties(height=260),
            width="stretch",
        )
    tr = book.trades()
    if not tr.empty:
        st.subheader("Closed trades")
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

    # ------------------------------------------------------------ capital
    with st.expander("Capital (paper)"):
        if st_ is None:
            cap = st.number_input(
                "Starting capital ₹",
                min_value=10000.0,
                value=float(book.capital(c.cfg.paper.live.capital)),
                step=50000.0,
            )
            if st.button("Save starting capital"):
                book.reset(cap)
                st.success(
                    f"Starting capital set to ₹{cap:,.0f}. The account opens at the next end-of-day update."
                )
        else:
            amt = st.number_input("Add (+) or withdraw (−) cash ₹", value=0.0, step=10000.0)
            if st.button("Apply cash change") and amt != 0:
                try:
                    book.move_cash(float(amt), "manual")
                    st.success(f"Cash {'added' if amt > 0 else 'withdrawn'}: ₹{abs(amt):,.0f}")
                except ValueError as e:
                    st.error(str(e))
            st.divider()
            new_cap = st.number_input(
                "Reset: new starting capital ₹",
                min_value=10000.0,
                value=float(c.cfg.paper.live.capital),
                step=50000.0,
            )
            sure = st.checkbox("I understand this deletes the paper history")
            if st.button("Reset paper account", disabled=not sure):
                book.reset(new_cap)
                st.success("Reset. The account re-opens at the next end-of-day update.")
