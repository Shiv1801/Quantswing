"""Performance metrics (05 §4) from the daily equity curve and the trade log."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

ANN = 252


def drawdown(equity: pd.Series) -> pd.Series:
    """``equity / running peak − 1``."""
    out: pd.Series = equity / equity.cummax() - 1.0
    return out


def max_dd_duration(equity: pd.Series) -> int:
    """Longest run of sessions below a prior peak."""
    under = (drawdown(equity) < 0).astype(int).to_numpy()
    best = run = 0
    for u in under:
        run = run + 1 if u else 0
        best = max(best, run)
    return best


def portfolio_metrics(
    daily: pd.DataFrame, bench: pd.Series | None, rf_annual: float
) -> dict[str, float]:
    """CAGR, vol, Sharpe, Sortino, Calmar, max DD (+duration), exposure, alpha/beta/IR."""
    eq = daily["equity"].astype("float64")
    r = eq.pct_change().dropna()
    n = len(r)
    if n < 2:
        return {}
    years = n / ANN
    growth = float(eq.to_numpy()[-1] / eq.to_numpy()[0])
    cagr = growth ** (1 / years) - 1 if years > 0 else float("nan")
    rf_d = (1 + rf_annual) ** (1 / ANN) - 1
    ex = r - rf_d
    vol = r.std() * math.sqrt(ANN)
    sharpe = ex.mean() / r.std() * math.sqrt(ANN) if r.std() > 0 else float("nan")
    downside = r[r < 0].std() * math.sqrt(ANN)
    sortino = ex.mean() * ANN / downside if downside > 0 else float("nan")
    mdd = float(drawdown(eq).min())
    out = {
        "cagr": float(cagr),
        "vol": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": mdd,
        "max_dd_sessions": float(max_dd_duration(eq)),
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else float("nan"),
        "exposure": float((daily["invested"] / eq).mean()),
        "avg_positions": float(daily["positions"].mean()),
        "halted_sessions": float(daily["halted"].sum()) if "halted" in daily else 0.0,
        "sessions": float(n),
    }
    if bench is not None:
        rb = bench.reindex(eq.index).pct_change().reindex(r.index)
        j = pd.concat([r, rb], axis=1).dropna().to_numpy(dtype="float64")
        if len(j) > 20 and j[:, 1].var(ddof=1) > 0:
            beta = float(np.cov(j[:, 0], j[:, 1])[0, 1] / j[:, 1].var(ddof=1))
            resid = j[:, 0] - beta * j[:, 1]
            out["beta"] = beta
            out["alpha_ann"] = float(resid.mean() * ANN)
            out["info_ratio"] = (
                float(resid.mean() / resid.std(ddof=1) * math.sqrt(ANN))
                if resid.std(ddof=1) > 0
                else float("nan")
            )
    return out


def trade_metrics(trades: pd.DataFrame) -> dict[str, float]:
    """Win rate, avg win/loss (R), expectancy (R), profit factor, capture, holding, turnover."""
    if trades.empty:
        return {"trades": 0.0}
    r = trades["r_mult"].astype(float)
    pnl = trades["net_pnl"].astype(float)
    wins, losses = r[r > 0], r[r <= 0]
    gw, gl = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
    p = len(wins) / len(r)
    return {
        "trades": float(len(r)),
        "win_rate": float(p),
        "avg_win_r": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_r": float(losses.mean()) if len(losses) else 0.0,
        "win_loss_ratio": (
            float(wins.mean() / abs(losses.mean()))
            if len(wins) and len(losses) and losses.mean() != 0
            else float("nan")
        ),
        "expectancy_r": float(r.mean()),
        "profit_factor": float(gw / gl) if gl > 0 else float("inf"),
        "capture": (
            float(trades["capture"].dropna().median())
            if trades["capture"].notna().any()
            else float("nan")
        ),
        "avg_holding": float(trades["holding"].mean()),
        "total_costs": float(trades["costs"].sum()),
        "net_pnl": float(pnl.sum()),
    }


def pnl_by_year(trades: pd.DataFrame) -> pd.Series:
    """Net P&L per exit year."""
    if trades.empty:
        return pd.Series(dtype="float64")
    return trades.groupby(pd.DatetimeIndex(trades["exit_date"]).year)["net_pnl"].sum()


def max_year_share(trades: pd.DataFrame) -> float:
    """Largest single-year share of total positive P&L (05 §3 regime split)."""
    y = pnl_by_year(trades)
    total = y[y > 0].sum()
    return float(y.max() / total) if total > 0 else float("nan")


def returns_moments(daily: pd.DataFrame) -> tuple[float, float, float, int]:
    """(per-period Sharpe, skew, kurtosis (non-excess), T) of daily returns — for DSR."""
    r = daily["equity"].pct_change().dropna()
    if len(r) < 3 or r.std() == 0:
        return float("nan"), 0.0, 3.0, len(r)
    arr = r.to_numpy(dtype="float64")
    return (
        float(arr.mean() / arr.std(ddof=1)),
        float(np.asarray(r.skew())),
        float(np.asarray(r.kurt())) + 3.0,
        len(r),
    )


def nan_to_none(d: dict[str, float | None]) -> dict[str, float | None]:
    """JSON-safe copy."""
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()}
