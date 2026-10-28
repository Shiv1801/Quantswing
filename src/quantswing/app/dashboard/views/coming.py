"""Placeholders for screens that need later phases (06 §4)."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from quantswing.app.dashboard.ui import Ctx

PLANNED: dict[str, tuple[str, list[str]]] = {
    "Watchlist": (
        "P4/P6",
        ["Triggers, SL, EV, p(win)", "Distance to trigger", "Live price via WebSocket"],
    ),
    "Positions": ("P6", ["Live R, MFE/MAE", "SL / trailing SL", "Days held, sleeve, exit plan"]),
    "Orders & Journal": ("P6", ["Order state machine log", "Fills, slippage", "Notes and tags"]),
    "Risk": (
        "P7",
        [
            "Exposure by sector (when available)",
            "Beta, correlation heatmap",
            "CVaR, stress scenarios",
        ],
    ),
}


def make(title: str) -> Callable[[Ctx], None]:
    """Build a placeholder renderer."""
    phase, items = PLANNED[title]

    def render(c: Ctx) -> None:
        st.title(title)
        st.info(f"Arrives in {phase}.")
        st.markdown("\n".join(f"- {x}" for x in items))

    return render
