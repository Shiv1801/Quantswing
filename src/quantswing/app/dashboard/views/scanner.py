"""Screen 2 — Scanner: whole universe ranked by composite, factor breakdown, filters."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from quantswing.alpha.research import COMPOSITE
from quantswing.app.dashboard import cache, data
from quantswing.app.dashboard.ui import Ctx, muted, styled

DEFAULT_COLS = [
    "TQ120",
    "MOM126",
    "RMOM",
    "H52",
    "KAL_T",
    "CMP",
    "RNG",
    "HURST",
    "ILLIQ",
    "VOLZ",
    "DD",
]


def render(c: Ctx) -> None:
    """Render the screen."""
    st.title("Scanner")
    snap = cache.snapshot(c.root_key, c.paths)
    if snap.empty:
        st.info(
            "No factor snapshot yet. Run `python scripts\\run_factors.py` and `python scripts\\p2_ic_report.py`."
        )
        return
    dates = sorted(snap["date"].unique(), reverse=True)
    f = st.columns([1.2, 1, 1, 2, 3])
    day = f[0].selectbox("Session", dates, format_func=lambda d: f"{pd.Timestamp(d):%d %b %Y}")
    gated = f[1].toggle("Gate pass only", value=True)
    all_factors = sorted(set(snap["factor"]) - {COMPOSITE, "GATE"})

    tbl = data.scanner_table(snap, pd.Timestamp(day))
    # The composite file can lag a session or two behind the regular factor snapshot
    # (they're rebuilt separately), so whether composite exists is checked per selected
    # session (via tbl), not against the whole multi-session snapshot (which may have it
    # for an older date only) - otherwise the filter below crashes with a KeyError.
    has_comp = COMPOSITE in tbl.columns
    min_score = f[2].number_input("Min composite", value=0.0, step=0.25, disabled=not has_comp)
    search = f[3].text_input("Search symbol / name", "")
    cols = f[4].multiselect(
        "Factors", all_factors, default=[x for x in DEFAULT_COLS if x in all_factors]
    )

    uni = cache.universe(c.root_key, c.paths)
    if not uni.empty:
        tbl = tbl.join(uni.set_index("symbol")[["name"]], how="left")
    if gated and "GATE" in tbl:
        tbl = tbl[tbl["GATE"]]
    if has_comp:
        tbl = tbl[tbl[COMPOSITE].fillna(-99) >= min_score]
    else:
        muted("Composite score not built yet for this session — showing raw factors only.")
    if search:
        q = search.upper()
        mask = tbl.index.str.upper().str.contains(q, regex=False)
        if "name" in tbl:
            mask |= tbl["name"].fillna("").str.upper().str.contains(q, regex=False)
        tbl = tbl[mask]
    show = (["name"] if "name" in tbl else []) + ([COMPOSITE] if has_comp else []) + cols
    view = tbl[show]
    muted(
        f"{len(view):,} stocks · neutralised cross-sectional z-scores (green = high) · sorted by composite"
    )
    st.dataframe(styled(view, [x for x in show if x != "name"]), width="stretch", height=640)
    st.download_button(
        "Download CSV",
        view.to_csv().encode(),
        f"scanner_{pd.Timestamp(day):%Y%m%d}.csv",
        "text/csv",
    )
