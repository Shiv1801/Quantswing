"""Slippage model (04 §2).

``slip_frac = half_spread + η · σ_daily · √(Q / ADV_shares)``;
buys fill at ``px·(1+slip)``, sells at ``px·(1−slip)``. ``multiplier`` scales both terms
(cost-stress run). Gap rule: a stop gapped through fills at the open (the engine passes
the open as ``px``).
"""

from __future__ import annotations

import math

from quantswing.core.config import SlippageCfg


class SlippageModel:
    """Spread + square-root impact."""

    def __init__(self, cfg: SlippageCfg, multiplier: float = 1.0) -> None:
        self.c = cfg
        self.m = multiplier

    def frac(self, qty: float, adv_shares: float, sigma_daily: float) -> float:
        """Fractional price concession for a fill of ``qty`` shares."""
        impact = 0.0
        if adv_shares > 0 and math.isfinite(sigma_daily) and sigma_daily > 0:
            impact = self.c.impact_eta * sigma_daily * math.sqrt(abs(qty) / adv_shares)
        return (self.c.default_half_spread_pct + impact) * self.m

    def price(
        self, side: str, px: float, qty: float, adv_shares: float, sigma_daily: float
    ) -> float:
        """Fill price after slippage."""
        s = self.frac(qty, adv_shares, sigma_daily)
        return px * (1 + s) if side == "buy" else px * (1 - s)
