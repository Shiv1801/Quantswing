"""Final round (D35): sector cap, take-profit (+ exact resume), 2× switch hurdle, vol level."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd

from quantswing.portfolio.construction import Controls, PortfolioSim, SimState, run_portfolio
from quantswing.portfolio.sizing import sector_capped
from tests.portfolio.test_v2 import _cfg, _inputs

EDGE = dict(selection="edge", weighting="conviction", max_names=15, min_weight=0.05, max_weight=0.2)


def test_sector_capped_respects_both_caps() -> None:
    raw = np.array([5.0, 4.0, 3.0, 1.0, 1.0, 1.0, 1.0])
    lab = np.array(["A", "A", "A", "B", "C", "Unknown", "D"])
    w = sector_capped(raw, lab, 1.0, 0.25, 0.30)
    assert w[lab == "A"].sum() <= 0.30 + 1e-9
    assert w.max() <= 0.25 + 1e-9 and w.sum() <= 1.0 + 1e-9
    assert w[lab != "A"].sum() > 0.5  # excess moved to other sectors


def test_sector_cap_in_decide() -> None:
    p, s, g = _inputs()
    cols = p.close.columns
    p = replace(
        p, sector=pd.Series(["Bank" if k < 20 else f"S{k}" for k in range(len(cols))], index=cols)
    )
    cfg = _cfg(**EDGE, sector_cap=0.30)
    cfg = cfg.model_copy(  # type: ignore[attr-defined]
        update={
            "factors": cfg.factors.model_copy(  # type: ignore[attr-defined]
                update={"xsec": cfg.factors.xsec.model_copy(update={"min_sector_names": 1})}  # type: ignore[attr-defined]
            )
        }
    )
    sim = PortfolioSim(p, s, g, cfg)
    w = sim.decide(200, np.zeros(len(cols), bool), 0.0, False, Controls())
    assert w[:20].sum() <= 0.30 + 1e-9 and w.sum() > 0.3


def test_take_profit_exits_and_exact_resume() -> None:
    p, s, g = _inputs()
    cfg = _cfg(**EDGE, take_profit=0.05, reentry_cooldown=20)
    idx = p.close.index
    full = run_portfolio(p, s, g, cfg, start=idx[100])
    tr = full.trades
    assert (tr["reason"] == "target").sum() > 0
    sim = PortfolioSim(p, s, g, cfg)
    st = sim.init_state(idx[100])
    eq: list[float] = []
    for a, b in ((100, 170), (171, 171), (172, len(idx) - 1)):
        st, d, _ = sim.run(st, a, b)
        st = SimState.from_json(json.loads(json.dumps(st.to_json())))
        eq += [x["equity"] for x in d]
    assert np.allclose(eq, full.daily["equity"].to_numpy(), atol=1e-6)


def test_higher_switch_hurdle_trades_less() -> None:
    p, s, g = _inputs()
    idx = p.close.index
    a = run_portfolio(p, s, g, _cfg(**EDGE, edge_ic=0.05), start=idx[100]).trades
    b = run_portfolio(
        p, s, g, _cfg(**EDGE, edge_ic=0.05, switch_cost_mult=5.0), start=idx[100]
    ).trades
    assert len(b) <= len(a)


def test_vol_target_level_scales_down() -> None:
    p, s, g = _inputs()
    held = np.zeros(p.close.shape[1], bool)
    hi = PortfolioSim(p, s, g, _cfg(**EDGE, vol_target=True, vol_target_level=5.0))
    lo = PortfolioSim(p, s, g, _cfg(**EDGE, vol_target=True, vol_target_level=0.01))
    assert (
        lo.decide(200, held, 0.0, False, Controls()).sum()
        < hi.decide(200, held, 0.0, False, Controls()).sum()
    )


def test_ic_exposure_series_no_lookahead_and_bounds() -> None:
    from quantswing.alpha.research import ic_exposure_series

    idx = pd.bdate_range("2020-01-01", periods=300)
    ic = pd.Series(np.r_[np.full(150, 0.1), np.full(150, -0.1)], index=idx)
    ic = ic + np.random.default_rng(0).normal(0, 0.02, 300)
    e = ic_exposure_series(ic, 10, 63, 2.0)
    assert e.between(0, 1).all() and (e.iloc[:50] == 1.0).all()  # 1 until enough history
    assert e.iloc[140] == 1.0 and e.iloc[-1] == 0.0
    assert e.iloc[155] == 1.0  # the negative ICs are not known yet (10-day lag)
    ic2 = ic.copy()
    ic2.iloc[200:] = 9.0  # changing the future must not change the past
    assert (ic_exposure_series(ic2, 10, 63, 2.0).iloc[:210] == e.iloc[:210]).all()


def test_exposure_scales_weights() -> None:
    p, s, g = _inputs()
    held = np.zeros(p.close.shape[1], bool)
    half = pd.Series(0.5, index=p.close.index)
    full = PortfolioSim(p, s, g, _cfg(**EDGE)).decide(200, held, 0.0, False, Controls())
    w = PortfolioSim(p, s, g, _cfg(**EDGE), exposure=half).decide(200, held, 0.0, False, Controls())
    assert np.allclose(w, full * 0.5)
