"""Screen 8 — Backtest Lab: compare runs, equity vs benchmark, drawdown, trades, R distribution."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from quantswing.app.dashboard import cache
from quantswing.app.dashboard.ui import Ctx, muted, styled

KEY_METRICS = [
    "cagr",
    "sharpe",
    "sortino",
    "max_dd",
    "calmar",
    "trades",
    "win_rate",
    "expectancy_r",
    "profit_factor",
    "capture",
    "avg_holding",
    "stress_expectancy_r",
]


def render(c: Ctx) -> None:
    """Render the screen."""
    st.title("Backtest Lab")
    runs = cache.runs(c.root_key, c.paths)
    if runs.empty:
        st.info("No runs yet. Run `python scripts\\run_backtest.py`.")
        return
    cols = ["run_id", "name"] + [m for m in KEY_METRICS if m in runs]
    st.subheader("Runs")
    pct = {"cagr", "max_dd", "win_rate", "capture"}
    colcfg = {
        m: st.column_config.NumberColumn(m, format="percent" if m in pct else "%.2f")
        for m in cols[2:]
    }
    colcfg["trades"] = st.column_config.NumberColumn("trades", format="%d")
    st.dataframe(
        runs[cols],
        width="stretch",
        height=min(35 * (len(runs) + 1), 300),
        hide_index=True,
        column_config=colcfg,
    )
    muted("Every run counts as a trial for the Deflated Sharpe (05 §3).")

    run_id = st.selectbox("Run", list(runs["run_id"]))
    daily = cache.run_artifact(c.root_key, c.paths, run_id, "daily")
    trades = cache.run_artifact(c.root_key, c.paths, run_id, "trades")
    if daily.empty:
        st.warning("This run has no saved equity curve.")
        return
    eq = daily["equity"].astype(float)
    eq.index = pd.to_datetime(eq.index)
    bench = cache.benchmark(c.root_key, c.paths, c.cfg.factors.beta.benchmark)
    curves = pd.DataFrame({"Strategy": eq / eq.iloc[0]})
    if not bench.empty:
        b = bench.reindex(eq.index).ffill()
        curves["Nifty 500"] = b / b.dropna().iloc[0]
    long = curves.reset_index(names="date").melt("date", var_name="series", value_name="growth")
    eq_ch = (
        alt.Chart(long)
        .mark_line(strokeWidth=1.4)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("growth:Q", title="Growth of ₹1", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=["Strategy", "Nifty 500"], range=["#4c9aff", "#8b93a1"]),
            ),
            tooltip=["date:T", "series:N", alt.Tooltip("growth:Q", format=".3f")],
        )
        .properties(height=280)
    )
    dd = (eq / eq.cummax() - 1).rename("dd").rename_axis("date").reset_index()
    dd_ch = (
        alt.Chart(dd)
        .mark_area(color="#d24b4b", opacity=0.5)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("dd:Q", title="Drawdown", axis=alt.Axis(format="%")),
        )
        .properties(height=110)
    )
    st.altair_chart(alt.vconcat(eq_ch, dd_ch).resolve_scale(x="shared"), width="stretch")

    if trades.empty or "r_mult" not in trades:
        if not trades.empty:
            st.subheader("Rebalance trades")
            key = next((k for k in ("date", "exit_date", "entry_date") if k in trades), None)
            st.dataframe(
                trades.sort_values(key, ascending=False) if key else trades,
                width="stretch",
                height=380,
                hide_index=True,
            )
        return
    left, mid, right = st.columns([2, 1, 1])
    with left:
        st.subheader("R-multiple distribution")
        hist = (
            alt.Chart(trades)
            .mark_bar(color="#4c9aff")
            .encode(
                x=alt.X("r_mult:Q", bin=alt.Bin(step=0.25), title="R"),
                y=alt.Y("count():Q", title="Trades"),
            )
            .properties(height=220)
        )
        st.altair_chart(hist, width="stretch")
    with mid:
        st.subheader("Exit reasons")
        st.dataframe(trades["reason"].value_counts().rename("trades"), width="stretch")
    with right:
        st.subheader("P&L by year")
        by_y = (
            trades.groupby(pd.DatetimeIndex(trades["exit_date"]).year)["net_pnl"]
            .sum()
            .rename("net ₹")
        )
        st.dataframe(
            styled(by_y.to_frame(), scale=float(by_y.abs().max() or 1), fmt="{:,.0f}"),
            width="stretch",
        )
    st.subheader("Trades")
    show = [
        "symbol",
        "entry_date",
        "exit_date",
        "entry_px",
        "exit_px",
        "qty",
        "r_mult",
        "mfe_r",
        "mae_r",
        "holding",
        "reason",
        "net_pnl",
    ]
    st.dataframe(
        styled(
            trades[show].sort_values("exit_date", ascending=False),
            ["r_mult", "mfe_r", "mae_r"],
            scale=2.0,
        ),
        width="stretch",
        height=420,
        hide_index=True,
    )
