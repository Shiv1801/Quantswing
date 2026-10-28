"""D39 cross-asset spill-over (Fyers-only drivers: indices, sector indices, India VIX, gold ETF).

Every driver is turned into a daily *driver return* ``d``:

* market-cap / sector indices → excess log return over the Nifty 500 (what the driver adds
  beyond the market);
* ``SMALL`` → Smallcap 250 minus Nifty 50 (the size spread), ``MID`` → Midcap 150 minus
  Nifty 50;
* ``VIX`` → change of ln(India VIX); ``GOLD`` / ``GSEC`` → plain log return.

A stock's *sensitivity* to a driver is the rolling OLS slope of its excess return
(``r_i − r_N500``) on ``d`` (``beta_window`` sessions). A *spill-over* factor is
sensitivity × the driver's move over the last ``move_window`` sessions — the part of that
move the stock "should" still reflect if prices catch up slowly (lead-lag).

Factors (each must pass the IC significance gate to enter the score; the model decides):

* ``MAC_SIZE``  — size spread;  ``MAC_VIX`` — volatility;  ``MAC_GOLD`` — gold;
* ``MAC_RATES`` — 10-year G-Sec index (if the Fyers symbol master has it);
* ``MAC_SECTOR`` — the sector index the stock tracks most closely (highest rolling
  correlation, point-in-time) × that sector's excess move.

:func:`exposures` also returns the per-stock market and best-sector sensitivities for the
trade engine's in-trade protection (D39 M3).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantswing.core.config import MacroCfg
from quantswing.factors.common import Panels, log_df

BROAD = ("N50", "MID", "SMALL", "VIX", "GOLD", "GSEC")  # not sector drivers
SPREADS = {"SMALL": "N50", "MID": "N50"}


def driver_returns(p: Panels) -> pd.DataFrame:
    """Daily driver returns (see module docstring). Empty frame without macro data."""
    if p.macro is None or p.macro.empty:
        return pd.DataFrame(index=p.close.index)
    lr = log_df(p.macro).diff()
    rb = p.rb
    out: dict[str, pd.Series] = {}
    for name in lr.columns:
        s = lr[name]
        if name in SPREADS and SPREADS[name] in lr:
            out[name] = s - lr[SPREADS[name]]
        elif name in ("VIX", "GOLD", "GSEC"):
            out[name] = s
        elif name == "N50":
            continue  # the spread base; the market itself is the Nifty 500
        else:
            out[name] = s - rb
    return pd.DataFrame(out, index=p.close.index)


def _excess(p: Panels) -> pd.DataFrame:
    out: pd.DataFrame = p.r.sub(p.rb, axis=0)
    return out


def rolling_beta(y: pd.DataFrame, x: pd.Series, window: int, min_obs: int) -> pd.DataFrame:
    """Rolling OLS slope of every column of ``y`` on ``x`` (pairwise complete)."""
    return rolling_beta_corr(y, x, window, min_obs)[0]


def rolling_beta_corr(
    y: pd.DataFrame, x: pd.Series, window: int, min_obs: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling (OLS slope, Pearson correlation) of every column of ``y`` on ``x``."""
    ok = y.notna() & x.notna().to_numpy()[:, None]
    yv = y.where(ok)
    xv = pd.DataFrame(np.where(ok, x.to_numpy()[:, None], np.nan), index=y.index, columns=y.columns)

    def m(a: pd.DataFrame) -> pd.DataFrame:
        out: pd.DataFrame = a.rolling(window, min_periods=min_obs).mean()
        return out

    mx, my = m(xv), m(yv)
    cov = m(xv * yv) - mx * my
    vx = m(xv * xv) - mx * mx
    vy = m(yv * yv) - my * my
    beta: pd.DataFrame = cov / vx.where(vx > 0)
    den = (vx * vy).where((vx > 0) & (vy > 0)) ** 0.5
    corr: pd.DataFrame = cov / den
    return beta, corr


@dataclass
class Exposures:
    """Per-stock sensitivities used by the trade engine (all known at each close)."""

    beta_mkt: pd.DataFrame  # slope on the Nifty 500 return
    beta_sec: pd.DataFrame  # slope on the best-matched sector's excess return
    dsec: pd.DataFrame  # that sector's excess return, mapped to each stock (per day)


def _best_sector(
    ex: pd.DataFrame, d: pd.DataFrame, mc: MacroCfg
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(β, driver return, driver move over ``move_window``) of each stock's closest sector."""
    shape = ex.shape
    best_c = np.full(shape, -np.inf)
    beta = np.full(shape, np.nan)
    dret = np.full(shape, np.nan)
    move = np.full(shape, np.nan)
    for name in d.columns:
        if name in BROAD:
            continue
        x = d[name]
        bb, cc = rolling_beta_corr(ex, x, mc.beta_window, mc.min_obs)
        c, b = cc.to_numpy(), bb.to_numpy()
        mv = x.rolling(mc.move_window, min_periods=mc.move_window).sum().to_numpy()[:, None]
        take = np.isfinite(c) & (c > best_c)
        best_c = np.where(take, c, best_c)
        beta = np.where(take, b, beta)
        dret = np.where(take, x.to_numpy()[:, None], dret)
        move = np.where(take, mv, move)

    def f(a: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(a, index=ex.index, columns=ex.columns)

    return f(beta), f(dret), f(move)


def compute(p: Panels, mc: MacroCfg) -> dict[str, pd.DataFrame]:
    """Spill-over factors (only for drivers that are loaded)."""
    out: dict[str, pd.DataFrame] = {}
    if not mc.enabled:
        return out
    d = driver_returns(p)
    if d.empty or d.shape[1] == 0:
        return out
    ex = _excess(p)
    live = p.close.notna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for name, fac in (
            ("SMALL", "MAC_SIZE"),
            ("VIX", "MAC_VIX"),
            ("GOLD", "MAC_GOLD"),
            ("GSEC", "MAC_RATES"),
        ):
            if name not in d:
                continue
            b = rolling_beta(ex, d[name], mc.beta_window, mc.min_obs)
            mv = d[name].rolling(mc.move_window, min_periods=mc.move_window).sum()
            out[fac] = b.mul(mv, axis=0).where(live)
        if any(c not in BROAD for c in d.columns):
            b, _, mv_s = _best_sector(ex, d, mc)
            out["MAC_SECTOR"] = (b * mv_s).where(live)
    return out


def exposures(p: Panels, mc: MacroCfg) -> Exposures | None:
    """Market + best-sector sensitivities for the in-trade protection rule."""
    d = driver_returns(p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bm = rolling_beta(p.r, p.rb, mc.beta_window, mc.min_obs)
        if d.empty or not any(c not in BROAD for c in d.columns):
            z = pd.DataFrame(0.0, index=p.close.index, columns=p.close.columns)
            return Exposures(bm, z, z)
        bs, dr, _ = _best_sector(_excess(p), d, mc)
    return Exposures(bm, bs, dr)
