"""Factor registry: one place that maps config → factor functions (03 §2).

``compute_all`` yields ``(name, raw_panel)`` one family at a time so the caller can
process and persist each factor before the next is built (bounded memory).

NSE factors (D31) — DELZ, RSEC, PLEDGE, EARN — are built only when their inputs are loaded.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from quantswing.core.config import FactorsCfg
from quantswing.factors import (
    drawdown,
    hurst,
    kalman,
    liquidity,
    macro,
    market,
    momentum,
    nse_factors,
    ou,
    trend,
    volatility,
)
from quantswing.factors.common import Panels

FAMILIES: tuple[str, ...] = (
    "trend",
    "momentum",
    "volatility",
    "liquidity",
    "hurst",
    "ou",
    "kalman",
    "drawdown",
    "market",
    "nse",
    "macro",
)


def compute_family(family: str, p: Panels, fc: FactorsCfg) -> dict[str, pd.DataFrame]:
    """Raw factor panels for one family."""
    if family == "trend":
        return trend.compute(p, fc.trend.windows, fc.annualisation_days)
    if family == "momentum":
        return momentum.compute(p, fc.momentum.lookbacks, fc.momentum.skip)
    if family == "volatility":
        c, rg = fc.compression, fc.range
        return volatility.compute(p, c.short, c.long, c.rank_window, rg.window, rg.vol_window)
    if family == "liquidity":
        return liquidity.compute(p, fc.illiquidity.window, fc.volume_surprise.window, fc.crore)
    if family == "hurst":
        return hurst.compute(p, fc.hurst.window, fc.hurst.q)
    if family == "ou":
        return ou.compute(p, fc.deviation.window, fc.ou_half_life.window)
    if family == "kalman":
        return kalman.compute(p, fc.kalman)
    if family == "drawdown":
        return drawdown.compute(p, fc.high52.window)
    if family == "market":
        rm = fc.residual_momentum
        return market.compute(p, fc.beta.window, rm.window, rm.skip)
    if family == "nse":
        return nse_factors.compute(p, fc)
    if family == "macro":
        return macro.compute(p, fc.macro)
    raise ValueError(f"unknown factor family {family!r}")


def compute_all(
    p: Panels, fc: FactorsCfg, skip: tuple[str, ...] = ()
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield ``(factor_name, raw_panel)`` for every factor (families in ``skip`` left out)."""
    for fam in FAMILIES:
        if fam not in skip:
            yield from compute_family(fam, p, fc).items()
