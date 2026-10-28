"""Screen 9 — Data Health: backfill progress (live), validation, adjustments, API usage."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from quantswing.app.dashboard import cache, data
from quantswing.app.dashboard.ui import Ctx, fmt_dt, muted


@st.fragment(run_every="15s")
def _progress(c: Ctx) -> None:
    s = data.status(c.paths)  # uncached: live while the backfill runs
    th = c.cfg.settings.throttle
    k = st.columns(5)
    k[0].metric("Stocks backfilled", f"{s.eq_done:,} / {s.universe:,}")
    k[1].metric("Failed", f"{s.eq_failed:,}")
    k[2].metric("Indices", f"{s.idx_done}")
    k[3].metric(
        "Fyers calls today",
        f"{s.api_today:,}",
        f"limit {int(th.per_day * th.safety_factor):,}",
        delta_color="off",
        delta_arrow="off",
    )
    k[4].metric("Last update", s.last_update[11:19] if s.last_update else "—")
    st.progress(min(s.eq_done / s.universe, 1.0) if s.universe else 0.0)
    muted("Refreshes every 15 s — safe to keep open while backfill.py runs.")


@st.fragment(run_every="30s")
def _nse(c: Ctx) -> None:
    """NSE secondary data (D31): cached files (live) + table sizes."""
    st.subheader("NSE data")
    root = c.paths.nse
    n = {
        k: len(list((root / k).glob("*/*.parquet"))) if (root / k).exists() else 0
        for k in ("bhav", "mto", "surv")
    }
    t = data.query(
        c.paths,
        "SELECT (SELECT count(*) FROM delisted) AS delisted, "
        "(SELECT count(*) FROM results_events) AS events, "
        "(SELECT count(DISTINCT symbol) FROM sectors) AS sectors, "
        "(SELECT count(*) FROM dividends) AS dividends",
    )
    k = st.columns(6)
    k[0].metric("Bhavcopy days", f"{n['bhav']:,}")
    k[1].metric("Delivery days", f"{n['mto']:,}")
    k[2].metric("Surveillance days", f"{n['surv']:,}")
    if not t.empty:
        k[3].metric("Delisted stocks", f"{int(t['delisted'].iloc[0]):,}")
        k[4].metric("Results events", f"{int(t['events'].iloc[0]):,}")
        k[5].metric("Sectors", f"{int(t['sectors'].iloc[0]):,}")
    muted("Counts update live while nse_backfill.py runs; tables after its final step.")


def render(c: Ctx) -> None:
    """Render the screen."""
    st.title("Data Health")
    _progress(c)
    if c.cfg.nse.enabled:
        _nse(c)

    bs = cache.backfill_state(c.root_key, c.paths)
    if not bs.empty and "error" in bs and bs["error"].notna().any():
        st.subheader("Failed symbols")
        st.dataframe(
            bs[bs["error"].notna()][["kind", "symbol", "error"]],
            width="stretch",
            hide_index=True,
            height=200,
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Validation issues")
        iss = cache.issues(c.root_key, c.paths)
        if not c.paths.issues.exists():
            muted("Available after the backfill's validation step.")
        elif iss.empty:
            st.success("Validation ran: no issues.")
        else:
            cnt = iss.groupby(["check", "severity"]).size().rename("rows").reset_index()
            st.dataframe(cnt, width="stretch", hide_index=True)
            q = iss[iss["severity"] == "quarantine"]
            if not q.empty:
                st.caption("Quarantined rows (most recent first)")
                st.dataframe(
                    q.sort_values("date", ascending=False).head(200),
                    width="stretch",
                    hide_index=True,
                    height=260,
                )
    with right:
        st.subheader("Split/bonus auto-adjustments")
        adj = cache.adjustments(c.root_key, c.paths)
        if not c.paths.adjustments.exists():
            muted("Available after the backfill's validation step.")
        elif adj.empty:
            st.success("None needed: Fyers prices look split/bonus-adjusted.")
        else:
            st.dataframe(
                adj.sort_values("ex_date", ascending=False),
                width="stretch",
                hide_index=True,
                height=300,
            )
            muted(
                "ratio = matched split/bonus factor applied to earlier prices; observed = actual close ratio."
            )

    st.subheader("Files")
    p = c.paths
    files = {
        "Universe": p.universe,
        "Calendar": p.calendar,
        "Validation issues": p.issues,
        "Database": p.duckdb,
        "Factor snapshot": p.factors / "snapshot.parquet",
        "IC report": p.reports / "p2_ic_report.html",
        "Backtest report": p.reports / "p3_backtest_report.html",
        "Fyers token": p.token,
    }
    st.dataframe(
        pd.DataFrame(
            [
                {"item": k, "updated": fmt_dt(data.mtime(v)), "path": str(v)}
                for k, v in files.items()
            ]
        ),
        width="stretch",
        hide_index=True,
    )
