"""Streamlit-cached wrappers around :mod:`data` (keyed by the data location)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from quantswing.app.dashboard import data
from quantswing.store.repositories import DataPaths

TTL_FAST = 15  # seconds — status/progress
TTL_SLOW = 300  # seconds — heavy tables


@st.cache_data(ttl=TTL_FAST, show_spinner=False)
def status(key: str, _p: DataPaths) -> data.Status:
    return data.status(_p)


@st.cache_data(ttl=TTL_FAST, show_spinner=False)
def backfill_state(key: str, _p: DataPaths) -> pd.DataFrame:
    return data.backfill_state(_p)


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def snapshot(key: str, _p: DataPaths) -> pd.DataFrame:
    return data.snapshot(_p)


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def universe(key: str, _p: DataPaths) -> pd.DataFrame:
    return data.universe(_p)


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def issues(key: str, _p: DataPaths) -> pd.DataFrame:
    return data.issues(_p)


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def adjustments(key: str, _p: DataPaths) -> pd.DataFrame:
    return data.adjustments(_p)


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def price_history(key: str, _p: DataPaths, symbol: str) -> pd.DataFrame:
    return data.price_history(_p, symbol)


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def factor_history(key: str, _p: DataPaths, symbol: str, factors: tuple[str, ...]) -> pd.DataFrame:
    return data.factor_history(_p, symbol, list(factors))


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def ic_results(key: str, _p: DataPaths) -> dict[str, pd.DataFrame]:
    return data.ic_results(_p)


@st.cache_data(ttl=TTL_FAST, show_spinner=False)
def runs(key: str, _p: DataPaths) -> pd.DataFrame:
    return data.runs(_p)


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def run_artifact(key: str, _p: DataPaths, run_id: str, name: str) -> pd.DataFrame:
    return data.run_artifact(_p, run_id, name)


@st.cache_data(ttl=TTL_SLOW, show_spinner=False)
def benchmark(key: str, _p: DataPaths, symbol: str) -> pd.Series:
    return data.benchmark(_p, symbol)
