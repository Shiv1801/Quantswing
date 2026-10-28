"""Factor library: hand-computed checks + point-in-time (no look-ahead) for every factor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantswing.core.config import get_config
from quantswing.factors import drawdown, hurst, kalman, liquidity, market, momentum, ou, trend, xsec
from quantswing.factors.common import Panels, rolling_time_regression
from quantswing.factors.gate import universe_gate
from quantswing.factors.registry import compute_all

FC = get_config().factors


def synth_panels(t: int = 420, n: int = 12, seed: int = 0) -> Panels:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=t, name="date")
    cols = [f"S{i}" for i in range(n)]
    rb = rng.normal(0.0003, 0.01, t)
    beta = rng.uniform(0.5, 1.5, n)
    r = rb[:, None] * beta + rng.normal(0, 0.015, (t, n))
    close = 100 * np.exp(np.cumsum(r, axis=0))
    opn = close * np.exp(rng.normal(0, 0.003, (t, n)))
    high = np.maximum(opn, close) * (1 + rng.uniform(0, 0.01, (t, n)))
    low = np.minimum(opn, close) * (1 - rng.uniform(0, 0.01, (t, n)))
    vol = rng.integers(10_000, 1_000_000, (t, n)).astype(float)
    df = lambda a: pd.DataFrame(a, index=idx, columns=cols)  # noqa: E731
    c = df(close)
    c.iloc[:40, 0] = np.nan  # late listing
    c.iloc[200:205, 1] = np.nan  # suspension
    mask = c.notna()
    return Panels(
        open=df(opn).where(mask),
        high=df(high).where(mask),
        low=df(low).where(mask),
        close=c,
        volume=df(vol).where(mask),
        turnover=(c * df(vol)).where(mask),
        bench=pd.Series(1000 * np.exp(np.cumsum(rb)), index=idx),
    )


def frame(values: list[float] | np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"A": values}, index=pd.bdate_range("2021-01-01", periods=len(values)))


def test_time_regression_exact_line() -> None:
    y = frame(np.log(100) + 0.002 * np.arange(200))
    lr = rolling_time_regression(y, 60)
    assert lr.slope["A"].iloc[-1] == pytest.approx(0.002, rel=1e-6)
    assert lr.r2["A"].iloc[-1] == pytest.approx(1.0, abs=1e-6)
    assert lr.fitted_end["A"].iloc[-1] == pytest.approx(y["A"].iloc[-1], abs=1e-8)
    assert np.isnan(lr.slope["A"].iloc[58])


def test_trend_quality_value() -> None:
    y = frame(np.log(100) + 0.001 * np.arange(200))
    tq = trend.trend_quality(y, 60, 252)
    assert tq["A"].iloc[-1] == pytest.approx(np.exp(0.252) - 1, rel=1e-5)


def test_momentum_value() -> None:
    lp = frame(np.log(100) + 0.001 * np.arange(300) + 0.01 * np.sin(np.arange(300)))
    r = lp.diff()
    m = momentum.vol_adjusted_momentum(lp, r, 63, 21)
    t = 299
    expect = (lp["A"].iloc[t - 21] - lp["A"].iloc[t - 63]) / (
        r["A"].iloc[t - 62 : t + 1].std() * np.sqrt(63)
    )
    assert m["A"].iloc[t] == pytest.approx(expect, rel=1e-9)


def test_drawdown_and_recovery() -> None:
    dd, rr = drawdown.drawdown_recovery(frame([100.0, 120.0, 90.0, 105.0, np.nan, 130.0]))
    assert list(dd["A"].round(4).fillna(9)) == [0, 0, -0.25, -0.125, 9, 0]
    assert list(rr["A"].round(4).fillna(9)) == [1, 1, 0, 0.5, 9, 1]


def test_ou_half_life_recovers_beta() -> None:
    rng = np.random.default_rng(3)
    x = np.zeros(3000)
    for i in range(1, 3000):
        x[i] = 0.8 * x[i - 1] + rng.normal()
    hl = ou.ou_half_life(frame(x), 2000)
    assert hl["A"].iloc[-1] == pytest.approx(-np.log(2) / np.log(0.8), rel=0.15)


def test_hurst_random_walk_near_half() -> None:
    rng = np.random.default_rng(4)
    r = frame(rng.normal(0, 0.01, 3000))
    h = hurst.variance_ratio_hurst(r, 2500, 10)
    assert h["A"].iloc[-1] == pytest.approx(0.5, abs=0.07)


def test_kalman_tracks_slope() -> None:
    lp = frame(np.log(100) + 0.001 * np.arange(600))
    s, t = kalman.kalman_trend(lp, FC.kalman)
    assert s["A"].iloc[-1] == pytest.approx(0.001, rel=0.05)
    assert t["A"].iloc[-1] > 0


def test_beta_ivol_exact() -> None:
    rng = np.random.default_rng(5)
    rb = pd.Series(rng.normal(0, 0.01, 400), index=pd.bdate_range("2021-01-01", periods=400))
    r = pd.DataFrame({"A": 2 * rb})
    b, iv = market.beta_ivol(r, rb, 252)
    assert b["A"].iloc[-1] == pytest.approx(2.0, rel=1e-9)
    assert iv["A"].iloc[-1] == pytest.approx(0.0, abs=1e-8)


def test_amihud_and_volume_surprise() -> None:
    r = frame([0.01] * 30)
    tv = frame([1e7] * 30)  # 1 crore
    assert liquidity.amihud(r, tv, 20, 1e7)["A"].iloc[-1] == pytest.approx(-np.log(0.01))
    v = frame([*np.exp(np.random.default_rng(6).normal(10, 0.1, 60)), np.exp(11.0)])
    assert liquidity.volume_surprise(v, 50)["A"].iloc[-1] > 5


def test_gate_rules() -> None:
    p = synth_panels()
    g = universe_gate(p, FC.gate)
    assert not g.iloc[:249].any().any()  # history
    assert g["S2"].iloc[300]
    assert not g["S0"].iloc[250]  # listed late
    assert not g["S1"].iloc[202]  # not trading that day
    locked = p.high.copy()
    locked.iloc[380:383, 3] = p.low.iloc[380:383, 3]  # 3-day one-price streak
    p2 = Panels(p.open, locked, p.low, p.close, p.volume, p.turnover, p.bench)
    g2 = universe_gate(p2, FC.gate)
    assert not g2["S3"].iloc[385] and g2["S3"].iloc[410]


def test_xsec_neutralisation() -> None:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2021-01-01", periods=5)
    size = pd.DataFrame(rng.normal(size=(5, 200)), index=idx)
    raw = size * 2 + pd.DataFrame(rng.normal(size=(5, 200)), index=idx)
    gate = pd.DataFrame(True, index=idx, columns=raw.columns)
    z, nz = xsec.process(raw, gate, size, FC.xsec)
    assert abs(z.iloc[0].corr(size.iloc[0])) > 0.8
    assert abs(nz.iloc[0].corr(size.iloc[0])) < 1e-8
    assert nz.iloc[0].mean() == pytest.approx(0, abs=1e-9) and nz.iloc[0].std() == pytest.approx(1)


def test_xsec_min_names() -> None:
    idx = pd.bdate_range("2021-01-01", periods=2)
    raw = pd.DataFrame(np.arange(20.0).reshape(2, 10), index=idx)
    gate = pd.DataFrame(True, index=idx, columns=raw.columns)
    z, _ = xsec.process(raw, gate, None, FC.xsec)
    assert z.isna().all().all()  # 10 names < min_names_per_date


def _eq(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return bool(((a - b).abs().le(1e-9) | (a.isna() & b.isna())).all().all())


def test_no_lookahead_every_factor() -> None:
    """Shift test (08 §4): altering data after t0 must not change any factor value ≤ t0."""
    p = synth_panels()
    t0 = 330
    rng = np.random.default_rng(11)
    shock = pd.DataFrame(
        np.exp(rng.normal(0, 0.2, p.close.shape)), index=p.close.index, columns=p.close.columns
    )
    shock.iloc[: t0 + 1] = 1.0
    bshock = pd.Series(np.exp(rng.normal(0, 0.2, len(p.bench))), index=p.bench.index)
    bshock.iloc[: t0 + 1] = 1.0
    wide = shock.copy()
    wide.iloc[t0 + 1 :] *= 1.05
    narrow = shock.copy()
    narrow.iloc[t0 + 1 :] *= 0.95
    q = Panels(
        p.open * shock,
        p.high * wide,
        p.low * narrow,
        p.close * shock,
        p.volume * shock,
        p.turnover * shock,
        p.bench * bshock,
    )
    base = dict(compute_all(p, FC))
    alt = dict(compute_all(q, FC))
    assert len(base) >= 18
    for name in base:
        assert _eq(base[name].iloc[: t0 + 1], alt[name].iloc[: t0 + 1]), name
    g1, g2 = universe_gate(p, FC.gate), universe_gate(q, FC.gate)
    assert g1.iloc[: t0 + 1].equals(g2.iloc[: t0 + 1])
