"""Screen 1 — Command Center: pipeline status, latest backtest, top scores."""

from __future__ import annotations

import streamlit as st

from quantswing.alpha.research import COMPOSITE
from quantswing.app.dashboard import cache, data
from quantswing.app.dashboard.ui import Ctx, fmt_dt, muted, styled


def _pct(a: float, b: float) -> str:
    return f"{a / b:.0%}" if b else "—"


def render(c: Ctx) -> None:
    """Render the screen."""
    st.title("Command Center")
    s = cache.status(c.root_key, c.paths)
    th = c.cfg.settings.throttle
    day_limit = int(th.per_day * th.safety_factor)
    k = st.columns(6)
    k[0].metric("Last session", f"{s.last_session:%d %b %Y}" if s.last_session else "—")
    k[1].metric("Universe", f"{s.universe:,}")
    k[2].metric(
        "Backfilled",
        f"{s.eq_done:,}",
        _pct(s.eq_done, s.universe),
        delta_color="off",
        delta_arrow="off",
    )
    k[3].metric(
        "History sessions",
        f"{s.sessions:,}",
        f"from {s.first_session:%Y}" if s.first_session else None,
        delta_color="off",
        delta_arrow="off",
    )
    k[4].metric(
        "Fyers calls today",
        f"{s.api_today:,}",
        f"of {day_limit:,}",
        delta_color="off",
        delta_arrow="off",
    )
    from quantswing.live.book import LiveBook
    from quantswing.portfolio.p5 import load_regime

    reg = load_regime(c.paths)
    st_now = reg["state"].dropna() if reg is not None else None
    if reg is not None and st_now is not None and len(st_now):
        k[5].metric(
            "Regime",
            str(st_now.iloc[-1]).replace("_", " "),
            f"exposure ×{reg['mult'].dropna().iloc[-1]:.1f}",
            delta_color="off",
            delta_arrow="off",
        )
    else:
        k[5].metric("Regime", "—", "run_portfolio.py", delta_color="off", delta_arrow="off")

    st.subheader("Pipeline")
    runs = cache.runs(c.root_key, c.paths)
    d37_runs = runs[runs["name"] == "swing_T24"] if not runs.empty and "name" in runs else runs
    live_book = LiveBook(c.paths.live_db)
    live_started = live_book.started
    steps = [
        (
            "P1 backfill",
            s.eq_done > 0 and s.eq_done >= 0.95 * max(s.universe, 1),
            f"{s.eq_done:,}/{s.universe:,} stocks · last {(s.last_update or '—')[:16].replace('T', ' ')}",
        ),
        ("P1 database", s.db_built is not None, f"built {fmt_dt(s.db_built)}"),
        ("P2 factors", s.factors_built is not None, f"built {fmt_dt(s.factors_built)}"),
        ("P2 IC report", bool(data.ic_results(c.paths)), "reports/p2_ic_report.html"),
        ("P3 baseline", not d37_runs.empty, f"{len(d37_runs)} T24 run(s) logged"),
        (
            "P4–P6",
            live_started,
            "live paper trading (T24)" if live_started else "run the EOD update to open the paper account",
        ),
    ]
    cols = st.columns(len(steps))
    for col, (name, ok, detail) in zip(cols, steps, strict=True):
        col.markdown(f"**{'✅' if ok else '⏳'} {name}**")
        col.caption(detail)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Top composite scores")
        snap = cache.snapshot(c.root_key, c.paths)
        tbl = data.scanner_table(snap)
        if tbl.empty or COMPOSITE not in tbl:
            muted("Available after run_factors.py + p2_ic_report.py.")
        else:
            top = (tbl[tbl["GATE"]] if "GATE" in tbl else tbl).head(15)
            show = [x for x in [COMPOSITE, "TQ120", "MOM126", "H52", "KAL_T", "ILLIQ"] if x in top]
            st.dataframe(styled(top[show]), width="stretch", height=420)
            muted(f"Session {snap['date'].max():%d %b %Y} · neutralised z-scores · gated universe")
    with right:
        st.subheader("Latest backtest (T24)")
        if d37_runs.empty:
            muted("Available after scripts/run_swing.py.")
        else:
            r = d37_runs.iloc[0]
            m = st.columns(3)
            m[0].metric("CAGR", f"{r.get('cagr', float('nan')):.1%}")
            m[1].metric("Sharpe", f"{r.get('sharpe', float('nan')):.2f}")
            m[2].metric("Max DD", f"{r.get('max_dd', float('nan')):.1%}")
            m = st.columns(3)
            m[0].metric("Trades", f"{r.get('trades', 0):,.0f}")
            m[1].metric("Expectancy", f"{r.get('expectancy_r', float('nan')):+.2f}R")
            m[2].metric("Profit factor", f"{r.get('profit_factor', float('nan')):.2f}")
            muted(f"Run {r['run_id']} · {r.get('name', '')}")
