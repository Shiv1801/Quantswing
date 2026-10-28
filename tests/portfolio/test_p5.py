"""P5: regime HMM (causal filter, labelling, smoothing), sizing, portfolio backtest, e2e."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantswing.core.config import get_config
from quantswing.factors.common import Panels
from quantswing.portfolio.construction import run_portfolio
from quantswing.portfolio.sizing import capped_normalise, vol_scale
from quantswing.regime.breadth import breadth
from quantswing.regime.hmm import filtered_probs, fit, run_regime, smooth_states
from tests.factors.test_factors import synth_panels

_BASE = get_config()
# D29 settings (regime + vol target + inverse-vol) for the regime/sizing tests; D33 has its own test
CFG = _BASE.model_copy(
    update={
        "risk": _BASE.risk.model_copy(
            update={
                "portfolio": _BASE.risk.portfolio.model_copy(
                    update={
                        "top_n": 12,
                        "weighting": "inverse_vol",
                        "vol_target": True,
                        "use_regime": True,
                        "universe_top_n": 0,
                    }
                )
            }
        )
    }
)
RC = CFG.risk.regime


def two_regime_features(n: int = 1600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2017-01-01", periods=n)
    calm = (np.arange(n) // 200) % 2 == 0
    ret = np.where(calm, rng.normal(0.001, 0.007, n), rng.normal(-0.002, 0.025, n))
    rv = pd.Series(ret).rolling(20, min_periods=1).std().to_numpy() * np.sqrt(252)
    br = np.where(calm, 0.7, 0.3) + rng.normal(0, 0.05, n)
    dv = np.where(calm, -0.01, 0.05) + rng.normal(0, 0.05, n)
    return pd.DataFrame({"ret": ret, "rv20": rv, "breadth": br, "dvix": dv}, index=idx)


def test_hmm_labels_and_causal_filter() -> None:
    f = two_regime_features()
    fitted = fit(f.to_numpy()[:1200], RC)
    assert set(fitted.labels) == {"bull_calm", "bull_volatile", "bear", "crisis"}
    x = f.to_numpy()
    p1 = filtered_probs(fitted, x)
    x2 = x.copy()
    x2[1400:] = x2[1400:] * 5  # change the future
    p2 = filtered_probs(fitted, x2)
    assert np.allclose(p1[:1400], p2[:1400])  # filtered = no look-ahead
    assert np.allclose(p1.sum(axis=1), 1.0)


def test_run_regime_separates_calm_and_stress() -> None:
    f = two_regime_features()
    reg = run_regime(f, RC, CFG.risk.regime_exposure_mult)
    live = reg.dropna(subset=["state"])
    calm = (np.arange(len(f)) // 200) % 2 == 0
    calm_s = pd.Series(calm, index=f.index).reindex(live.index)
    m_calm = live.loc[calm_s, "mult"].mean()
    m_stress = live.loc[~calm_s, "mult"].mean()
    assert m_calm > m_stress + 0.3
    assert reg["state"].iloc[: RC.min_train_sessions].isna().all()  # nothing before the first fit


def test_smooth_states_requires_confirmation() -> None:
    idx = pd.bdate_range("2024-01-01", periods=6)
    pr = pd.DataFrame(
        {"bull_calm": [0.9, 0.9, 0.2, 0.9, 0.2, 0.2], "bear": [0.1, 0.1, 0.8, 0.1, 0.8, 0.8]},
        index=idx,
    )
    s = smooth_states(pr, 0.6, 2)
    assert list(s) == ["bull_calm", "bull_calm", "bull_calm", "bull_calm", "bull_calm", "bear"]


def test_breadth() -> None:
    idx = pd.bdate_range("2024-01-01", periods=150)
    lp = pd.DataFrame(
        {"UP": np.linspace(0, 1, 150), "DN": np.linspace(1, 0, 150), "UP2": np.linspace(0, 2, 150)},
        index=idx,
    )
    g = pd.DataFrame(True, index=idx, columns=lp.columns)
    b = breadth(lp, g, 100, 2)
    assert b.iloc[-1] == pytest.approx(2 / 3) and b.iloc[50] != b.iloc[50]  # NaN before full window


def test_sizing() -> None:
    w = capped_normalise(np.array([10.0, 1.0, 1.0, 1.0]), 1.0, 0.4)
    assert w.sum() == pytest.approx(1.0) and w.max() <= 0.4 + 1e-12
    cov = np.eye(2) * (0.02**2)
    assert vol_scale(np.array([0.5, 0.5]), cov, 0.15) == pytest.approx(
        min(1.0, 0.15 / (0.02 * np.sqrt(0.5) * np.sqrt(252)))
    )


def _panels_with_edge(edge: float, seed: int = 3) -> tuple[Panels, pd.DataFrame, pd.DataFrame]:
    p = synth_panels(t=900, n=60, seed=seed)
    rng = np.random.default_rng(seed)
    score = pd.DataFrame(
        rng.normal(size=p.close.shape), index=p.close.index, columns=p.close.columns
    )
    # plant: tomorrow's return rises with today's score
    r = p.r.fillna(0.0) + edge * score.shift(1).fillna(0.0)
    close = 100 * np.exp(r.cumsum())
    close = close.where(p.close.notna())
    q = Panels(
        close.shift(1).fillna(100.0).where(p.close.notna()),
        close * 1.005,
        close * 0.995,
        close,
        p.volume,
        close * p.volume,
        p.bench,
    )
    gate = pd.DataFrame(True, index=p.close.index, columns=p.close.columns)
    return q, score, gate


def test_portfolio_accounting_and_regime_zero() -> None:
    p, score, gate = _panels_with_edge(0.0)
    res = run_portfolio(p, score * np.nan, gate, CFG)  # no scores → all cash
    y = (1 + CFG.costs.idle_cash_annual_yield) ** (1 / 252)
    assert res.trades.empty and res.daily["equity"].iloc[-1] == pytest.approx(
        CFG.risk.paper_capital_inr * y ** (len(res.daily) - 1)
    )
    off = run_portfolio(p, score, gate, CFG, regime_mult=pd.Series(0.0, index=p.close.index))
    assert off.trades.empty
    on = run_portfolio(p, score, gate, CFG)
    assert on.daily["positions"].max() <= CFG.risk.portfolio.top_n
    assert (on.trades.groupby("date")["symbol"].nunique() > 0).all()
    assert on.daily["invested"].div(on.daily["equity"]).max() <= 1.0 + 1e-9


def test_portfolio_captures_planted_edge() -> None:
    p, score, gate = _panels_with_edge(0.004)
    good = run_portfolio(p, score, gate, CFG)
    bad = run_portfolio(p, -score, gate, CFG)
    g = good.daily["equity"].iloc[-1]
    b = bad.daily["equity"].iloc[-1]
    assert g > CFG.risk.paper_capital_inr * 1.2 > b


def test_p5_end_to_end(tmp_path: Path) -> None:
    from quantswing.portfolio.p5 import load_regime, run_p5
    from quantswing.portfolio.report import write_report
    from quantswing.store.repositories import DataPaths
    from tests.sleeves.test_sleeves import build_long_root

    root = build_long_root(tmp_path)
    paths = DataPaths.relocated(root)
    rc = CFG.risk.model_copy(
        update={
            "regime": CFG.risk.regime.model_copy(
                update={"min_train_sessions": 300, "breadth_min_names": 20}
            )
        }
    )
    cfg = CFG.model_copy(update={"risk": rc})
    variants, bench, reg = run_p5(paths, cfg)
    assert [v.name for v in variants] == ["portfolio_no_regime", "portfolio_regime"]
    assert load_regime(paths) is not None and reg["state"].notna().any()
    assert all(set(v.checks) and v.run_id for v in variants)
    out = write_report(variants, bench, reg, None or pd.Series(dtype=float), paths.reports, ["n"])  # type: ignore[arg-type]
    assert "factor portfolio" in out.read_text(encoding="utf-8")


def test_d33_liquid_universe_slots_and_score_decides() -> None:
    p = synth_panels(300, 30)
    idx, cols = p.close.index, p.close.columns
    score = pd.DataFrame(-1.0, index=idx, columns=cols)
    score.iloc[:, :3] = 2.0  # only 3 names have a positive score
    gate = p.close.notna()
    pc = _BASE.risk.portfolio.model_copy(update={"universe_top_n": 20})
    cfg = _BASE.model_copy(update={"risk": _BASE.risk.model_copy(update={"portfolio": pc})})
    turn = p.turnover.copy()
    turn.iloc[:, 1] = 1.0  # name 1: least traded → outside the top-20 universe
    res = run_portfolio(replace(p, turnover=turn), score, gate, cfg, start=idx[100])
    held = set(res.trades["symbol"])
    assert cols[1] not in held and held <= {cols[0], cols[2]}
    first = res.trades[res.trades["date"] == res.trades["date"].min()]
    w = first["value"] / cfg.risk.paper_capital_inr
    assert np.allclose(w, 1 / pc.top_n, atol=0.01)  # a 10% slot each; the rest stays cash
    off = run_portfolio(p, score, gate, cfg, regime_mult=pd.Series(0.0, index=idx), start=idx[100])
    assert not off.trades.empty  # regime is display-only in D33
