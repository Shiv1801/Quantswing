"""Build any sleeve by name."""

from __future__ import annotations

import pandas as pd

from quantswing.alpha.events import event_block
from quantswing.core.config import AppConfig
from quantswing.factors.common import Panels
from quantswing.sleeves import (
    s1_breakout,
    s2_pullback,
    s3_ath_recovery,
    s4_high_momentum,
    s5_pead,
)
from quantswing.sleeves.base import FactorView, SleeveSignals

SLEEVES: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5")


def _raw(name: str, p: Panels, f: FactorView, gate: pd.DataFrame, cfg: AppConfig) -> SleeveSignals:
    r = cfg.sleeves.rules
    if name == "S1":
        return s1_breakout.build(p, f, gate, r.S1, cfg.backtest.exits.sigma_window)
    if name == "S2":
        return s2_pullback.build(p, f, gate, r.S2)
    if name == "S3":
        return s3_ath_recovery.build(p, f, gate, r.S3)
    if name == "S4":
        return s4_high_momentum.build(p, f, gate, r.S4)
    if name == "S5":
        return s5_pead.build(p, f, gate, r.S5)
    raise ValueError(f"unknown sleeve {name!r}")


def build(name: str, p: Panels, f: FactorView, gate: pd.DataFrame, cfg: AppConfig) -> SleeveSignals:
    """Signals for sleeve ``name`` (S1–S4 skip entries in the results window, D31)."""
    sig = _raw(name, p, f, gate, cfg)
    if name == "S5":
        return sig
    block = event_block(p, cfg.risk.event_window)
    if block is None:
        return sig
    s = sig.signal & ~block.reindex_like(sig.signal).fillna(False).astype(bool)
    return SleeveSignals(sig.name, s, sig.entry_level, sig.confirm)
