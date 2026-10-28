"""Screen 10 — Settings: current policy parameters (read-only in v1)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from quantswing.app.dashboard.ui import Ctx, muted


def _flat(d: dict[str, object], prefix: str = "") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out += _flat(v, key + ".")
        else:
            out.append((key, str(v)))
    return out


def render(c: Ctx) -> None:
    """Render the screen."""
    st.title("Settings")
    muted(
        "Read-only in v1. Edit the YAML files in config\\ and restart; every change is versioned (08 §3)."
    )
    tabs = st.tabs(["Risk", "Sleeves", "Backtest", "Factors", "Costs", "Data", "NSE"])
    sections = [
        c.cfg.risk.model_dump(mode="json"),
        c.cfg.sleeves.model_dump(mode="json"),
        c.cfg.backtest.model_dump(mode="json"),
        c.cfg.factors.model_dump(mode="json"),
        c.cfg.costs.model_dump(mode="json"),
        c.cfg.settings.data.model_dump(mode="json")
        | {"validation": c.cfg.settings.validation.model_dump(mode="json")},
        c.cfg.nse.model_dump(mode="json"),
    ]
    for tab, sec in zip(tabs, sections, strict=True):
        with tab:
            st.dataframe(
                pd.DataFrame(_flat(sec), columns=["parameter", "value"]),
                width="stretch",
                hide_index=True,
                height=520,
            )
    st.subheader("Kill switch")
    st.button("Flatten all / pause entries", disabled=True, help="Active from P6 (paper broker).")
