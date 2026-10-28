"""v2 options: conviction sizing bounds, cost-aware switching, drift band, trailing stop
(+ exact resume), significance-gated IC weights."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from quantswing.alpha.combine import ic_ir_weights
from quantswing.core.config import get_config
from quantswing.portfolio.construction import PortfolioSim, SimState, run_portfolio
from tests.factors.test_factors import synth_panels

CFG = get_config()


def _cfg(**kw: object) -> object:
    pc = CFG.risk.portfolio.model_copy(update={"dd_brake": False, "rebalance_band": 0.25, **kw})
    ew = CFG.risk.event_window.model_copy(update={"enabled": False})
    return CFG.model_copy(
        update={"risk": CFG.risk.model_copy(update={"portfolio": pc, "event_window": ew})}
    )


def _inputs(seed: int = 1) -> tuple[object, pd.DataFrame, pd.DataFrame]:
    p = synth_panels(400, 40, seed=3)
    s = pd.DataFrame(
        np.random.default_rng(seed).normal(0, 1, p.close.shape),
        index=p.close.index,
        columns=p.close.columns,
    )
    return p, s, p.close.notna()


def test_conviction_weights_within_bounds_and_fully_invested() -> None:
    p, s, g = _inputs()
    cfg = _cfg(
        selection="edge", weighting="conviction", max_names=15, min_weight=0.05, max_weight=0.2
    )
    sim = PortfolioSim(p, s, g, cfg)
    w = sim.decide(
        200,
        np.zeros(p.close.shape[1], bool),
        0.0,
        False,
        __import__("quantswing.portfolio.construction", fromlist=["Controls"]).Controls(),
    )
    pos = w[w > 0]
    assert 5 <= len(pos) <= 15 and pos.max() <= 0.2 + 1e-9 and pos.min() >= 0.05 - 1e-9
    assert abs(w.sum() - 1.0) < 1e-6


def test_edge_switching_trades_less_than_buffer() -> None:
    p, s, g = _inputs()
    idx = p.close.index
    a = run_portfolio(p, s, g, _cfg(hold_buffer_mult=1.0), start=idx[100]).trades
    b = run_portfolio(
        p, s, g, _cfg(selection="edge", max_names=10, edge_ic=0.05), start=idx[100]
    ).trades
    assert len(b) < len(a)  # a new name must beat a holding by more than the round-trip cost


def test_trailing_exits_and_exact_resume() -> None:
    p, s, g = _inputs()
    cfg = _cfg(
        selection="edge",
        weighting="conviction",
        trail_sigma=0.5,
        trail_horizon=10,
        reentry_cooldown=20,
    )
    idx = p.close.index
    full = run_portfolio(p, s, g, cfg, start=idx[100])
    assert (full.trades["reason"] == "trail").sum() > 0
    sim = PortfolioSim(p, s, g, cfg)
    st = sim.init_state(idx[100])
    eq: list[float] = []
    for a, b in ((100, 170), (171, 171), (172, len(idx) - 1)):
        st, d, _ = sim.run(st, a, b)
        st = SimState.from_json(json.loads(json.dumps(st.to_json())))
        eq += [x["equity"] for x in d]
    assert np.allclose(eq, full.daily["equity"].to_numpy(), atol=1e-6)


def test_significance_gate_drops_noise_factors() -> None:
    idx = pd.bdate_range("2018-01-01", periods=900)
    rng = np.random.default_rng(0)
    ic = pd.DataFrame(
        {"real": 0.05 + rng.normal(0, 0.05, 900), "noise": rng.normal(0, 0.05, 900)}, index=idx
    )
    cc = CFG.factors.ic.combine.model_copy(update={"min_t": 2.0, "sign_aware": True})
    w = ic_ir_weights(ic, 10, cc).dropna()
    assert (w["noise"].abs() < 1e-12).mean() > 0.8 and (w["real"] > 0).all()
