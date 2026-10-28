"""Shared factor primitives on wide panels (index = session date, columns = symbol).

Conventions (08 §3): pure functions, no I/O, trailing windows only (value at t uses data
≤ t), NaN where the full window is not available. ``r`` = daily log return.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Panels:
    """Adjusted daily inputs aligned on the session calendar.

    ``bench`` is the benchmark close (Series on the same index) for beta/residual factors.
    Optional NSE inputs (D31; ``None`` when not loaded):

    * ``deliv`` — delivery % (0–100) panel;
    * ``sector`` — symbol → NSE sector (current classification);
    * ``restricted`` — bool panel, under ASM/GSM/ESM/IRP/default/BZ that day;
    * ``pledge`` — 1.0 / 0.0 panel where the surveillance file exists (NaN before);
    * ``events`` — long table ``symbol, event_date`` (results published → first session);
    * ``meetings`` — long table ``symbol, meeting_session, known_session`` (results dates);
    * ``div_yield`` — dividend ÷ previous close on ex-dates (0 elsewhere);
    * ``delisted_last`` — symbol → last traded date for delisted stocks;
    * ``delisted_haircut`` — symbol → exit haircut (0 for mergers/schemes, NaN = config).
    """

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    turnover: pd.DataFrame
    bench: pd.Series
    deliv: pd.DataFrame | None = None
    sector: pd.Series | None = None
    restricted: pd.DataFrame | None = None
    pledge: pd.DataFrame | None = None
    events: pd.DataFrame | None = None
    meetings: pd.DataFrame | None = None
    div_yield: pd.DataFrame | None = None
    delisted_last: pd.Series | None = None
    delisted_haircut: pd.Series | None = None  # per delisted symbol (0 for mergers); NaN = config
    macro: pd.DataFrame | None = None  # D39: driver closes (index / ETF), columns = driver names

    @property
    def logp(self) -> pd.DataFrame:
        """ln(close)."""
        return pd.DataFrame(np.log(self.close), index=self.close.index, columns=self.close.columns)

    @property
    def r(self) -> pd.DataFrame:
        """Daily log returns ``ln(P_t / P_{t-1})``."""
        return self.logp.diff()

    @property
    def rb(self) -> pd.Series:
        """Benchmark daily log returns."""
        return pd.Series(np.log(self.bench), index=self.bench.index).diff()


def exp_df(x: pd.DataFrame) -> pd.DataFrame:
    """Element-wise exp keeping the DataFrame type."""
    return pd.DataFrame(np.exp(x.to_numpy(dtype="float64")), index=x.index, columns=x.columns)


def log_df(x: pd.DataFrame) -> pd.DataFrame:
    """Element-wise natural log (non-positive → NaN) keeping the DataFrame type."""
    v = x.to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(v > 0, np.log(np.where(v > 0, v, 1.0)), np.nan)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def rolling_sigma(r: pd.DataFrame, n: int) -> pd.DataFrame:
    """Std of daily log returns over the last ``n`` sessions (full window required)."""
    return r.rolling(n, min_periods=n).std()


@dataclass(frozen=True)
class LinReg:
    """Rolling OLS of y on time: slope ``b``, ``r2``, fitted value at window end, residual sd."""

    slope: pd.DataFrame
    r2: pd.DataFrame
    fitted_end: pd.DataFrame
    resid_sd: pd.DataFrame


def rolling_time_regression(y: pd.DataFrame, n: int) -> LinReg:
    """Regress ``y`` on t = 0..n-1 in each trailing window of ``n`` sessions.

    With centred time, ``var_t = (n²−1)/12`` is constant, so:
    ``b = cov(t, y) / var_t``, ``R² = cov² / (var_t · var_y)``,
    fitted at window end ``= mean_y + b·(n−1)/2``,
    residual sd ``= sqrt(var_y · (1−R²) · n/(n−2))``.
    Global time index is used for the cross moment; the window mean of t is subtracted.
    """
    t = pd.DataFrame(
        np.broadcast_to(np.arange(len(y), dtype="float64")[:, None], y.shape),
        index=y.index,
        columns=y.columns,
    )
    mean_y = y.rolling(n, min_periods=n).mean()
    mean_t = t.where(y.notna()).rolling(n, min_periods=n).mean()
    cov_ty = (t * y).rolling(n, min_periods=n).mean() - mean_t * mean_y
    var_t = (n * n - 1) / 12.0
    var_y = (y * y).rolling(n, min_periods=n).mean() - mean_y * mean_y
    var_y = var_y.clip(lower=0.0)
    slope = cov_ty / var_t
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = (cov_ty * cov_ty) / (var_t * var_y)
    r2 = r2.where(var_y > 0).clip(upper=1.0)
    fitted_end = mean_y + slope * (n - 1) / 2.0
    resid_sd = np.sqrt(var_y * (1.0 - r2) * n / (n - 2))
    return LinReg(slope, r2, fitted_end, pd.DataFrame(resid_sd, index=y.index, columns=y.columns))


def rolling_ar1_beta(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling AR(1) coefficient of ``x_t`` on ``x_{t−1}`` over ``n`` sessions."""
    x0, x1 = x.shift(1), x
    m0, m1 = x0.rolling(n, min_periods=n).mean(), x1.rolling(n, min_periods=n).mean()
    cov = (x0 * x1).rolling(n, min_periods=n).mean() - m0 * m1
    var0 = (x0 * x0).rolling(n, min_periods=n).mean() - m0 * m0
    out: pd.DataFrame = cov / var0.where(var0 > 0)
    return out


def run_length(flag: pd.DataFrame) -> pd.DataFrame:
    """Length of the current run of ``True`` values (0 when ``False``)."""
    f = flag.fillna(False).astype("int64")
    c = f.cumsum()
    reset = c.where(f == 0).ffill().fillna(0)
    return c - reset
