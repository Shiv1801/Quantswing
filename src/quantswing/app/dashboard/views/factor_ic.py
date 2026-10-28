"""Factor IC (P2 results): mean IC by horizon, verdicts, IC by year."""

from __future__ import annotations

import streamlit as st

from quantswing.app.dashboard import cache
from quantswing.app.dashboard.ui import Ctx, muted, styled


def render(c: Ctx) -> None:
    """Render the screen."""
    st.title("Factor IC")
    res = cache.ic_results(c.root_key, c.paths)
    if "stats" not in res:
        st.info("No IC results yet. Run `python scripts\\p2_ic_report.py`.")
        return
    stats = res["stats"]
    piv = stats.pivot(index="factor", columns="h", values="mean")
    piv.columns = [f"{h}d" for h in piv.columns]
    tpiv = stats.pivot(index="factor", columns="h", values="t_nonoverlap")
    tpiv.columns = [f"t {h}d" for h in tpiv.columns]
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Mean rank IC by horizon")
        st.dataframe(styled(piv, scale=0.05, fmt="{:+.3f}"), width="stretch", height=560)
        muted(
            "Rank IC of neutralised factor vs forward log return. |IC| ≈ 0.02–0.05 is useful in practice."
        )
    with right:
        if "verdicts" in res:
            st.subheader("Verdict (primary horizon)")
            v = res["verdicts"][["mean", "t_nonoverlap", "hit", "verdict"]]
            st.dataframe(
                styled(v, ["mean", "t_nonoverlap"], scale=3.0), width="stretch", height=560
            )
    st.subheader("t-stats (non-overlapping)")
    st.dataframe(styled(tpiv, scale=3.0, fmt="{:+.1f}"), width="stretch")
    if "by_year" in res:
        st.subheader("Mean IC by year (primary horizon)")
        st.dataframe(styled(res["by_year"], scale=0.08, fmt="{:+.3f}"), width="stretch")
    muted(f"Full HTML report: {c.paths.reports / 'p2_ic_report.html'}")
