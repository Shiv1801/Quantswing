"""Stock view: adjusted price, factor history and latest factor values for one symbol."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from quantswing.alpha.research import COMPOSITE
from quantswing.app.dashboard import cache, data
from quantswing.app.dashboard.ui import Ctx, muted, styled

DEFAULT = [COMPOSITE, "TQ120", "MOM126", "H52"]


def render(c: Ctx) -> None:
    """Render the screen."""
    st.title("Stock")
    uni = cache.universe(c.root_key, c.paths)
    if uni.empty:
        st.info("No universe yet — run the P1 backfill.")
        return
    snap = cache.snapshot(c.root_key, c.paths)
    tbl = data.scanner_table(snap)
    options = (
        list(tbl.index) + [s for s in uni["symbol"] if s not in set(tbl.index)]
        if not tbl.empty
        else list(uni["symbol"])
    )
    names = uni.set_index("symbol")["name"].to_dict()
    top = st.columns([2, 3, 1])
    sym = top[0].selectbox(
        "Symbol", options, format_func=lambda s: f"{s.split(':')[-1]} — {names.get(s, '')}"
    )
    all_f = sorted(set(snap["factor"]) - {"GATE"}) if not snap.empty else []
    chosen = top[1].multiselect("Factors", all_f, default=[x for x in DEFAULT if x in all_f])
    years = top[2].selectbox("Window", [1, 2, 3, 5, 10], index=2, format_func=lambda y: f"{y}y")

    px = cache.price_history(c.root_key, c.paths, sym)
    if px.empty:
        st.warning("No prices in the database for this symbol yet.")
        return
    px = px[px["date"] >= px["date"].max() - pd.DateOffset(years=years)]
    base = alt.Chart(px).encode(x=alt.X("date:T", title=None))
    line = base.mark_line(strokeWidth=1.4, color="#4c9aff").encode(
        y=alt.Y("close:Q", title="Adj. close", scale=alt.Scale(zero=False)),
        tooltip=[
            "date:T",
            alt.Tooltip("close:Q", format=",.2f"),
            alt.Tooltip("volume:Q", format=",.0f"),
        ],
    )
    q = (
        base.transform_filter("datum.quarantined")
        .mark_point(color="#e0a030", size=40)
        .encode(y="close:Q")
    )
    vol = base.mark_bar(color="#3a4150").encode(y=alt.Y("volume:Q", title="Volume"))
    st.altair_chart(
        alt.vconcat((line + q).properties(height=300), vol.properties(height=80)).resolve_scale(
            x="shared"
        ),
        width="stretch",
    )
    muted("Split/bonus-adjusted · orange points = quarantined rows (validation)")

    if chosen:
        fh = cache.factor_history(c.root_key, c.paths, sym, tuple(chosen))
        fh = fh[fh.index >= px["date"].min()]
        if not fh.empty:
            long = (
                fh.reset_index(names="date")
                .melt("date", var_name="factor", value_name="z")
                .dropna()
            )
            ch = (
                alt.Chart(long)
                .mark_line(strokeWidth=1.2)
                .encode(
                    x=alt.X("date:T", title=None),
                    y=alt.Y("z:Q", title="z"),
                    color="factor:N",
                    tooltip=["date:T", "factor:N", alt.Tooltip("z:Q", format="+.2f")],
                )
                .properties(height=220)
            )
            st.altair_chart(ch, width="stretch")
    if not tbl.empty and sym in tbl.index:
        st.subheader("Latest factor values")
        row = tbl.loc[[sym]].drop(columns=["GATE"], errors="ignore")
        st.dataframe(styled(row), width="stretch")
