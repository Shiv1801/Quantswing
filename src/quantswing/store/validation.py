"""Data validation checks (02 §3, adapted for Fyers-only data — D22).

All checks are pure: candle DataFrame (``CANDLE_SCHEMA`` columns) in → issue rows out.
Issue row: ``{symbol, date, check, severity, detail}``.

| Check | Rule | Severity |
|---|---|---|
| ``ohlc`` | NaN / non-positive price, or not ``L ≤ min(O,C) ≤ max(O,C) ≤ H`` | reject |
| ``volume`` | volume ≤ 0 on a traded day (equities) | warn |
| ``off_calendar`` | candle date inside the observed range but not a session | quarantine |
| ``ca_suspect`` | overnight gap ``|ln(O_t/C_{t−1})| > ca_gap_min_abs_log`` and ``C_t/C_{t−1}`` within ``ca_ratio_tolerance`` of a common split/bonus ratio → likely UNADJUSTED corporate action | quarantine |
| ``ca_adjusted`` | same as ``ca_suspect`` but auto back-adjusted (``ca_auto_adjust``); see :func:`detect_adjustments` | info |
| ``spike`` | ``|r_t| > spike_sigma · σ_{t−1}`` (rolling σ of log returns, window ``spike_vol_window``, excluding t) and not ``ca_suspect`` | quarantine |
| ``missing_sessions`` | sessions between the symbol's first and last candle with no candle | info |
| ``stale`` | last candle older than ``stale_max_sessions`` sessions | warn |
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from quantswing.core.calendar import TradingCalendar
from quantswing.core.config import KnownEventCfg, ValidationCfg
from quantswing.store.schema import ISSUE_COLUMNS

Issue = dict[str, Any]


def _issue(symbol: str, d: Any, check: str, severity: str, detail: str) -> Issue:
    return {"symbol": symbol, "date": d, "check": check, "severity": severity, "detail": detail}


def check_ohlc(df: pd.DataFrame, symbol: str) -> list[Issue]:
    """Reject rows with NaN/non-positive prices or inconsistent OHLC."""
    o, h, lo, c = df["open"], df["high"], df["low"], df["close"]
    bad_val = (
        o.isna() | h.isna() | lo.isna() | c.isna() | (o <= 0) | (h <= 0) | (lo <= 0) | (c <= 0)
    )
    body_lo = np.minimum(o, c)
    body_hi = np.maximum(o, c)
    bad_rel = (lo > body_lo) | (body_hi > h) | (lo > h)
    bad = bad_val | bad_rel
    return [
        _issue(symbol, d, "ohlc", "reject", f"O={r.open} H={r.high} L={r.low} C={r.close}")
        for d, r in zip(df.loc[bad, "date"], df.loc[bad].itertuples(), strict=True)
    ]


def check_volume(df: pd.DataFrame, symbol: str) -> list[Issue]:
    """Warn on non-positive volume (equities only)."""
    bad = df["volume"].fillna(0) <= 0
    return [_issue(symbol, d, "volume", "warn", "volume <= 0") for d in df.loc[bad, "date"]]


def check_calendar(df: pd.DataFrame, symbol: str, cal: TradingCalendar) -> list[Issue]:
    """Off-calendar candles (quarantine) + one ``missing_sessions`` info row per symbol."""
    out: list[Issue] = []
    if cal.first is None or df.empty:
        return out
    dates = list(df["date"])
    for d in dates:
        if cal.first <= d <= cal.last and d not in cal.sessions:
            out.append(_issue(symbol, d, "off_calendar", "quarantine", "not an observed session"))
    have = set(dates)
    missing = [d for d in cal.trading_days(dates[0], dates[-1]) if d not in have]
    if missing:
        out.append(
            _issue(
                symbol, missing[0], "missing_sessions", "info", f"{len(missing)} sessions missing"
            )
        )
    return out


def check_stale(
    df: pd.DataFrame, symbol: str, cal: TradingCalendar, latest: date, max_sessions: int
) -> list[Issue]:
    """Warn if the last candle is more than ``max_sessions`` sessions before ``latest``."""
    if df.empty:
        return [_issue(symbol, None, "stale", "warn", "no data")]
    last = df["date"].iloc[-1]
    gap = sum(1 for _ in cal.trading_days(last, latest)) - 1
    if gap > max_sessions:
        return [_issue(symbol, last, "stale", "warn", f"last candle {gap} sessions old")]
    return []


def ca_suspects(df: pd.DataFrame, cfg: ValidationCfg) -> pd.Series:
    """Boolean mask: rows that look like an unadjusted split/bonus (see module table)."""
    prev_c = df["close"].shift(1)
    gap = pd.Series(np.log(df["open"] / prev_c), index=df.index).abs()
    ratio = df["close"] / prev_c
    near = pd.Series(False, index=df.index)
    for cand in cfg.ca_candidate_ratios:
        near |= (ratio / cand - 1).abs() <= cfg.ca_ratio_tolerance
    out: pd.Series = (gap > cfg.ca_gap_min_abs_log) & near
    return out


def matched_ratio(observed: float, cfg: ValidationCfg) -> float | None:
    """Nearest candidate split/bonus ratio within ``ca_ratio_tolerance`` (else ``None``)."""
    best = min(cfg.ca_candidate_ratios, key=lambda c: abs(observed / c - 1))
    return best if abs(observed / best - 1) <= cfg.ca_ratio_tolerance else None


def detect_adjustments(df: pd.DataFrame, symbol: str, cfg: ValidationCfg) -> list[dict[str, Any]]:
    """Adjustment events for suspected unadjusted splits/bonuses.

    Each event: ``{symbol, ex_date, ratio, observed}`` where ``ratio`` is the matched
    candidate (e.g. 0.5 for a 1:1 bonus). Prices strictly before ``ex_date`` are later
    multiplied by ``ratio`` (volume divided) — standard back-adjustment.
    """
    if len(df) < 2:
        return []
    df = df.sort_values("date").reset_index(drop=True)
    ca = ca_suspects(df, cfg)
    ratio = df["close"] / df["close"].shift(1)
    out: list[dict[str, Any]] = []
    for i in df.index[ca]:
        m = matched_ratio(float(ratio[i]), cfg)
        if m is not None:
            out.append(
                {
                    "symbol": symbol,
                    "ex_date": df["date"].iloc[i],
                    "ratio": m,
                    "observed": float(ratio[i]),
                }
            )
    return out


def check_jumps(df: pd.DataFrame, symbol: str, cfg: ValidationCfg) -> list[Issue]:
    """``ca_suspect`` (quarantine, or ``ca_adjusted`` info when auto-adjusted) and ``spike``.

    Returns on suspect dates are excluded from the rolling σ so one split does not mask
    real spikes for the next ``spike_vol_window`` sessions.
    """
    if len(df) < 2:
        return []
    out: list[Issue] = []
    ca = ca_suspects(df, cfg)
    r = pd.Series(np.log(df["close"] / df["close"].shift(1)), index=df.index)
    sigma = r.mask(ca).rolling(cfg.spike_vol_window, min_periods=cfg.spike_min_obs).std().shift(1)
    spike = (r.abs() > cfg.spike_sigma * sigma) & ~ca
    ratio = df["close"] / df["close"].shift(1)
    for i in df.index[ca]:
        m = matched_ratio(float(ratio[i]), cfg)
        if cfg.ca_auto_adjust and m is not None:
            out.append(
                _issue(
                    symbol,
                    df["date"].iloc[i],
                    "ca_adjusted",
                    "info",
                    f"C/Cprev={ratio[i]:.4f} -> x{m}",
                )
            )
        else:
            out.append(
                _issue(
                    symbol,
                    df["date"].iloc[i],
                    "ca_suspect",
                    "quarantine",
                    f"C/Cprev={ratio[i]:.4f}",
                )
            )
    for i in df.index[spike.fillna(False)]:
        out.append(
            _issue(
                symbol,
                df["date"].iloc[i],
                "spike",
                "quarantine",
                f"r={r[i]:.4f} ({abs(r[i]) / sigma[i] if sigma[i] > 0 else float('inf'):.1f} sigma)",
            )
        )
    return out


def validate_symbol(
    df: pd.DataFrame,
    symbol: str,
    cfg: ValidationCfg,
    cal: TradingCalendar,
    latest: date,
    *,
    is_equity: bool = True,
) -> list[Issue]:
    """Run all per-symbol checks on a date-sorted candle frame."""
    df = df.sort_values("date").reset_index(drop=True)
    issues = check_ohlc(df, symbol)
    rejected = {i["date"] for i in issues}
    clean = df[~df["date"].isin(rejected)].reset_index(drop=True)
    if is_equity:
        issues += check_volume(clean, symbol)
    issues += check_calendar(clean, symbol, cal)
    issues += check_jumps(clean, symbol, cfg)
    issues += check_stale(clean, symbol, cal, latest, cfg.stale_max_sessions)
    return issues


def issues_frame(issues: Sequence[Issue]) -> pd.DataFrame:
    """Issue rows → DataFrame with ``ISSUE_COLUMNS``."""
    df = pd.DataFrame(list(issues), columns=list(ISSUE_COLUMNS))
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


ADJ_COLUMNS = ("symbol", "ex_date", "ratio", "observed")


def adjustments_frame(events: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Adjustment events → DataFrame (``symbol, ex_date, ratio, observed``)."""
    df = pd.DataFrame(list(events), columns=list(ADJ_COLUMNS))
    df["ex_date"] = pd.to_datetime(df["ex_date"]).dt.date
    df["ratio"] = df["ratio"].astype("float64")
    df["observed"] = df["observed"].astype("float64")
    return df


@dataclass(frozen=True)
class EventProbe:
    """Result of checking one known corporate action in Fyers data."""

    symbol: str
    ex_date: date
    expected_ratio: float
    observed_ratio: float | None
    verdict: str  # ADJUSTED / UNADJUSTED / INCONCLUSIVE / NO_DATA


def probe_known_event(df: pd.DataFrame | None, ev: KnownEventCfg, tol: float) -> EventProbe:
    """Is Fyers history adjusted for ``ev``? Compares ``C_ex / C_prev`` to 1 and to ``ratio``."""
    if df is None or df.empty:
        return EventProbe(ev.symbol, ev.ex_date, ev.ratio, None, "NO_DATA")
    before = df[df["date"] < ev.ex_date]
    on = df[df["date"] == ev.ex_date]
    if before.empty or on.empty:
        return EventProbe(ev.symbol, ev.ex_date, ev.ratio, None, "NO_DATA")
    obs = float(on["close"].iloc[0] / before["close"].iloc[-1])
    if abs(obs - 1.0) <= tol:
        verdict = "ADJUSTED"
    elif abs(obs / ev.ratio - 1.0) <= tol:
        verdict = "UNADJUSTED"
    else:
        verdict = "INCONCLUSIVE"
    return EventProbe(ev.symbol, ev.ex_date, ev.ratio, obs, verdict)
