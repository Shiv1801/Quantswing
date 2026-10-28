"""Sleeve framework (03 §4): each sleeve turns panels + factors into dated entry signals.

A sleeve produces, for every (session t, symbol):
* ``signal`` — True at the close of t ⇒ candidate entry on t+1;
* ``entry_level`` — stop-entry price for t+1 (NaN ⇒ market-on-open);
* ``confirm`` — condition that must hold on the entry day (NaN/None ⇒ none).
All inputs are known at the close of t (point-in-time); ``confirm`` uses the entry day's
full-session volume as a proxy for the projected intraday volume (04 §1), noted in D27.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from quantswing.factors.common import Panels
from quantswing.store.panels import FactorStore


@dataclass(frozen=True)
class SleeveSignals:
    """Output of a sleeve's rule set."""

    name: str
    signal: pd.DataFrame
    entry_level: pd.DataFrame | None = None
    confirm: pd.DataFrame | None = None

    def events(self) -> pd.DataFrame:
        """Long table of signal (date, symbol)."""
        s = self.signal.fillna(False).astype(bool)
        rows, cols = s.to_numpy().nonzero()
        return pd.DataFrame({"date": s.index[rows], "symbol": s.columns[cols]})


class MissingFactorError(KeyError):
    """A factor panel needed by a sleeve was not built (factors older than the code)."""

    def __str__(self) -> str:
        return f"factor {self.args[0]} not found - run: python scripts\\run_factors.py"


REQUIRED: dict[str, tuple[str, ...]] = {
    "S1": ("CMP", "RNG", "TQ120", "HURST", "VOLZ"),
    "S2": ("TQ120", "R2_120", "DEV", "OUHL", "VOLZ", "KAL_T"),
    "S3": ("DD", "RR", "HURST", "KAL_SLOPE", "KAL_T"),
    "S4": ("H52", "MOM126", "RMOM", "VOLZ"),
    "S5": ("BETA", "IVOL", "VOLZ"),
}


class FactorView:
    """Lazy access to raw factor panels aligned to the price panels."""

    def __init__(
        self,
        panels: Panels,
        store: FactorStore | None = None,
        preset: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        self.p = panels
        self.store = store
        self._cache: dict[str, pd.DataFrame] = dict(preset or {})

    def __getitem__(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            if self.store is None:
                raise KeyError(name)
            if not self.store.path("raw", name).exists():
                raise MissingFactorError(name)
            self._cache[name] = self.store.read("raw", name).reindex_like(self.p.close)
        return self._cache[name]


def xs_pct_rank(x: pd.DataFrame, gate: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cross-sectional percentile rank (0–1] per date, optionally within the gate."""
    y = x.where(gate.astype(bool)) if gate is not None else x
    return y.rank(axis=1, pct=True)


def rising_edge(cond: pd.DataFrame) -> pd.DataFrame:
    """True on the first day a condition becomes true."""
    c = cond.fillna(False).astype(bool)
    prev = c.shift(1)
    return c & ~prev.fillna(False).astype(bool)


SleeveFn = Callable[..., SleeveSignals]
