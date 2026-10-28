"""Shared UI helpers: context, caching, dense styling, colour scales."""

from __future__ import annotations

import os
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from quantswing.core.config import AppConfig, get_config
from quantswing.store.repositories import DataPaths

DENSE_CSS = """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 1rem; max-width: 100%;}
div[data-testid="stMetric"] {background: #181b21; border: 1px solid #262a33; border-radius: 6px; padding: 6px 10px;}
div[data-testid="stMetricValue"] {font-size: 1.25rem;}
div[data-testid="stMetricLabel"] p {font-size: 0.75rem; color: #9aa3b2;}
h1 {font-size: 1.35rem !important; margin-bottom: 0.2rem;}
h2, h3 {font-size: 1.0rem !important; margin: 0.6rem 0 0.2rem 0;}
.qs-muted {color: #8b93a1; font-size: 0.8rem;}
</style>
"""


@dataclass(frozen=True)
class Ctx:
    """Per-session context."""

    cfg: AppConfig
    paths: DataPaths
    root_key: str  # cache key for the data location


def ctx() -> Ctx:
    """Build the context. ``QS_DASHBOARD_DATA`` relocates all data (tests/demo)."""
    cfg = get_config()
    alt = os.environ.get("QS_DASHBOARD_DATA")
    paths = DataPaths.relocated(Path(alt), cfg) if alt else DataPaths(cfg)
    return Ctx(cfg, paths, str(paths.duckdb))


def setup_page() -> None:
    """Inject the dense stylesheet."""
    st.markdown(DENSE_CSS, unsafe_allow_html=True)


def muted(text: str) -> None:
    """Small grey caption."""
    st.markdown(f'<div class="qs-muted">{text}</div>', unsafe_allow_html=True)


def diverging(v: Any, scale: float = 2.0) -> str:
    """Green/red background for signed values (no matplotlib needed)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(x):
        return ""
    a = min(abs(x) / scale, 1.0) * 0.7
    rgb = "34,160,90" if x > 0 else "210,60,60"
    return f"background-color: rgba({rgb},{a:.2f})"


def styled(
    df: pd.DataFrame, cols: list[str] | None = None, scale: float = 2.0, fmt: str = "{:+.2f}"
) -> Any:
    """Styler with diverging colours on ``cols`` (default: all numeric)."""
    num = (
        cols
        if cols is not None
        else [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    )
    sub: list[Hashable] = list(num)
    sty = df.style.map(lambda v: diverging(v, scale), subset=sub)
    return sty.format(fmt, subset=sub, na_rep="")


def fmt_dt(x: Any) -> str:
    """Compact timestamp."""
    return "—" if x is None else f"{x:%d %b %H:%M}"
