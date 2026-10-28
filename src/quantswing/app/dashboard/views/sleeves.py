"""Sleeves: walk-forward verdicts, OOS curves, fitted live exits (P4)."""

from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from quantswing.app.dashboard import cache
from quantswing.app.dashboard.ui import Ctx, muted
from quantswing.sleeves.runner import fitted_path


def render(c: Ctx) -> None:
    """Render the screen."""
    st.title("Sleeves")
    f = fitted_path(c.paths)
    if not f.exists():
        st.info("No sleeve validation yet. Run `python scripts\\run_sleeves.py`.")
        return
    data = json.loads(f.read_text(encoding="utf-8"))
    sl = data.get("sleeves", {})
    cols = st.columns(max(len(sl), 1))
    for col, (name, s) in zip(cols, sl.items(), strict=False):
        oos = s.get("oos", {})
        col.metric(
            f"{name} · {s['verdict']}",
            f"Sharpe {oos.get('sharpe') or float('nan'):.2f}",
            f"{(oos.get('trades') or 0):.0f} OOS trades",
            delta_color="off",
            delta_arrow="off",
        )
    muted(
        f"Updated {data.get('updated', '—')} · verdict ON needs OOS Sharpe ≥ 0.7, ≥ 30 trades, 2×-cost expectancy > 0"
    )

    rows = []
    for name, s in sl.items():
        ex, oos = s.get("exits", {}), s.get("oos", {})
        rows.append(
            {
                "sleeve": name,
                "verdict": s["verdict"],
                "signals": s.get("signals"),
                "CAGR": oos.get("cagr"),
                "Sharpe": oos.get("sharpe"),
                "MaxDD": oos.get("max_dd"),
                "win": oos.get("win_rate"),
                "Exp R": oos.get("expectancy_r"),
                "PF": oos.get("profit_factor"),
                "2x Exp R": s.get("stress_expectancy_r"),
                "k_sl σ": ex.get("k_sl_sigma"),
                "partial R": ex.get("partial_at_r"),
                "trail k": ex.get("trail_k_below_switch"),
                "time stop": ex.get("time_stop_sessions"),
                "why OFF": "; ".join(s.get("reasons", [])),
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        column_config={
            "CAGR": st.column_config.NumberColumn(format="percent"),
            "MaxDD": st.column_config.NumberColumn(format="percent"),
            "win": st.column_config.NumberColumn(format="percent"),
            "Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "Exp R": st.column_config.NumberColumn(format="%+.2f"),
            "PF": st.column_config.NumberColumn(format="%.2f"),
            "2x Exp R": st.column_config.NumberColumn(format="%+.2f"),
            "k_sl σ": st.column_config.NumberColumn(format="%.2f"),
            "partial R": st.column_config.NumberColumn(format="%.2f"),
            "trail k": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    curves = []
    for name, s in sl.items():
        rid = s.get("run_id")
        d = cache.run_artifact(c.root_key, c.paths, rid, "daily") if rid else pd.DataFrame()
        if not d.empty:
            e = d["equity"].astype(float)
            curves.append(
                pd.DataFrame(
                    {
                        "date": pd.to_datetime(d.index),
                        "growth": (e / e.iloc[0]).to_numpy(),
                        "sleeve": name,
                    }
                )
            )
    if curves:
        st.subheader("Out-of-sample equity (test years chained)")
        ch = (
            alt.Chart(pd.concat(curves))
            .mark_line(strokeWidth=1.3)
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("growth:Q", title="Growth of ₹1", scale=alt.Scale(zero=False)),
                color="sleeve:N",
                tooltip=["date:T", "sleeve:N", alt.Tooltip("growth:Q", format=".3f")],
            )
            .properties(height=300)
        )
        st.altair_chart(ch, width="stretch")
    for name, s in sl.items():
        rid = s.get("run_id")
        folds = cache.run_artifact(c.root_key, c.paths, rid, "folds") if rid else pd.DataFrame()
        if not folds.empty:
            with st.expander(f"{name}: per-year folds (fitted on prior years)"):
                st.dataframe(folds, width="stretch", hide_index=True)
