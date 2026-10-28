"""QuantSwing dashboard (Streamlit v1, D13). Launch: ``python scripts/run_dashboard.py``."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from quantswing.app.dashboard.ui import Ctx, ctx, setup_page
from quantswing.app.dashboard.views import (
    backtest_lab,
    command_center,
    data_health,
    factor_ic,
    live,
    login,
    scanner,
    settings,
    sleeves,
    stock,
    trade_log,
)

st.set_page_config(page_title="QuantSwing", page_icon="📈", layout="wide")
setup_page()
C: Ctx = ctx()


def page(fn: Callable[[Ctx], None], title: str, icon: str, default: bool = False) -> st.Page:
    """Wrap a view as a navigation page."""

    def run() -> None:
        fn(C)

    run.__name__ = title.lower().replace(" ", "_").replace("&", "and")
    return st.Page(run, title=title, icon=icon, default=default)


nav = st.navigation(
    {
        "Trading": [
            page(live.render, "Actions", "✅", default=True),
            page(trade_log.render, "Trade Log", "📒"),
            page(login.render, "Login", "🔑"),
        ],
        "Overview": [page(command_center.render, "Command Center", "🛰️")],
        "Research": [
            page(scanner.render, "Scanner", "🔎"),
            page(stock.render, "Stock", "📉"),
            page(factor_ic.render, "Factor IC", "🧪"),
            page(backtest_lab.render, "Backtest Lab", "🧮"),
            page(sleeves.render, "Sleeves", "🧩"),
        ],
        "System": [
            page(data_health.render, "Data Health", "🩺"),
            page(settings.render, "Settings", "⚙️"),
        ],
    }
)
nav.run()
