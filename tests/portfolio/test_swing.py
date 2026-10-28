"""D37 trade engine: SL / gap / half-target + break-even / time exit, sizing, no look-ahead."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from quantswing.core.config import get_config
from quantswing.factors.common import Panels
from quantswing.portfolio.swing import SwingResult, run_swing
from tests.factors.test_factors import synth_panels

CFG = get_config()
S5 = "S5"


def _setup() -> tuple[Panels, pd.DataFrame, pd.DataFrame]:
    p = synth_panels(260, 12, seed=4)
    rng = np.random.default_rng(1)
    s = pd.DataFrame(
        rng.normal(0, 0.3, p.close.shape), index=p.close.index, columns=p.close.columns
    )
    s[S5] = 5.0  # always the top stock
    return p, s, p.close.notna()


def _run(p: Panels, s: pd.DataFrame, g: pd.DataFrame) -> SwingResult:
    return run_swing(p, s, g, CFG, start=p.close.index[80])


def _first(res: SwingResult) -> pd.Series:
    t = res.trades
    return t[t["symbol"] == S5].sort_values("entry_date").iloc[0]


def _set(p: Panels, day: pd.Timestamp, **px: float) -> Panels:
    out = {}
    for k in ("open", "high", "low", "close"):
        f = getattr(p, k).copy()
        if k in px:
            f.loc[day, S5] = px[k]
        out[k] = f
    return replace(p, **out)


def test_stop_loss_fills_at_the_stop() -> None:
    """Basic SL-fill mechanics - independent of whatever sl_grace_days the current
    config default happens to be (D37 vs T24), so pin it to 0 explicitly here."""
    cfg = _with_swing(sl_grace_days=0)
    p, s, g = _setup()
    tr = _first(run_swing(p, s, g, cfg, start=p.close.index[80]))
    day = p.close.index[p.close.index.get_loc(tr["entry_date"]) + 2]
    sl = tr["sl"]
    q = _set(p, day, open=sl * 1.02, high=sl * 1.03, low=sl * 0.97, close=sl * 0.98)
    t2 = _first(run_swing(q, s, g, cfg, start=p.close.index[80]))
    assert t2["exit_reason"] == "SL" and pd.Timestamp(t2["exit_date"]) == day
    assert sl * 0.99 < t2["exit_px"] <= sl  # at the stop, minus slippage
    assert -1.3 < t2["r_multiple"] < -0.9  # about −1R after costs


def _with_swing(**kw: object) -> object:
    sw = CFG.risk.swing.model_copy(update=kw)
    return CFG.model_copy(update={"risk": CFG.risk.model_copy(update={"swing": sw})})


def test_sl_grace_period_delays_the_stop() -> None:
    """D41: sl_grace_days=3 means a stop hit on day 2 is ignored (held < grace); the
    position keeps trading and can still exit later by other means."""
    p, s, g = _setup()
    tr = _first(_run(p, s, g))
    day = p.close.index[p.close.index.get_loc(tr["entry_date"]) + 2]  # held == 2 that day
    sl = tr["sl"]
    q = _set(p, day, open=sl * 1.02, high=sl * 1.03, low=sl * 0.97, close=sl * 0.98)
    cfg = _with_swing(sl_grace_days=3)
    t2 = _first(run_swing(q, s, g, cfg, start=p.close.index[80]))
    # the same low that triggered "SL" with no grace period must NOT close the trade here
    assert not (t2["exit_reason"] == "SL" and pd.Timestamp(t2["exit_date"]) == day)


def test_regime_gate_blocks_new_entries_only() -> None:
    """D41: when the regime multiplier is below the cutoff, no NEW positions open, but
    the engine (exits, existing holdings) otherwise runs as normal."""
    p, s, g = _setup()
    off = pd.Series(0.0, index=p.close.index)  # always below any positive cutoff
    cfg = _with_swing(regime_gate=True, regime_gate_mult=0.5)
    res = run_swing(p, s, g, cfg, start=p.close.index[80], regime_mult=off)
    assert res.trades.empty  # nothing ever opened
    on = pd.Series(1.0, index=p.close.index)  # always above the cutoff -> behaves like D37
    base = _run(p, s, g)
    gated = run_swing(p, s, g, cfg, start=p.close.index[80], regime_mult=on)
    assert len(gated.trades) == len(base.trades)


def test_gap_below_stop_fills_at_the_open() -> None:
    p, s, g = _setup()
    tr = _first(_run(p, s, g))
    day = p.close.index[p.close.index.get_loc(tr["entry_date"]) + 2]
    o = tr["sl"] * 0.90
    q = _set(p, day, open=o, high=o * 1.01, low=o * 0.98, close=o)
    t2 = _first(_run(q, s, g))
    assert t2["exit_reason"] == "SL" and o * 0.99 < t2["exit_px"] <= o


def test_target_sells_half_then_break_even() -> None:
    p, s, g = _setup()
    tr = _first(_run(p, s, g))
    i = p.close.index.get_loc(tr["entry_date"])
    d1, d2 = p.close.index[i + 2], p.close.index[i + 4]
    e, tg = tr["entry_px"], tr["target"]
    q = _set(p, d1, open=e * 1.01, high=tg * 1.01, low=e * 1.005, close=tg * 0.995)
    q = _set(q, d2, open=e * 1.01, high=e * 1.02, low=e * 0.98, close=e * 0.99)
    t2 = _first(_run(q, s, g))
    assert pd.Timestamp(t2["partial_date"]) == d1 and t2["partial_px"] == tg
    assert abs(t2["partial_qty"] - t2["qty"] / 2) <= 1
    assert t2["exit_reason"] in ("BE", "TSL") and d1 < pd.Timestamp(t2["exit_date"]) <= d2
    assert t2["pnl"] > 0  # half booked at the target, the rest protected (BE / TSL)


def test_time_exit_and_invariants() -> None:
    p, s, g = _setup()
    res = _run(p, s, g)
    t = res.trades
    assert (t.loc[t["exit_reason"] == "Time", "sessions"] == CFG.risk.swing.max_hold).all()
    assert (res.daily["cash"] > -1e-6).all()
    assert (res.daily["positions"] <= CFG.risk.swing.max_positions).all()
    top = max(f for _, f in CFG.risk.swing.risk_tiers)
    eq = res.daily["equity"].max()
    assert (t["risk_inr"] <= top * eq * 1.02).all()
    assert (t["invested"] <= CFG.risk.swing.max_weight * eq * 1.02).all()
    assert set(t["exit_reason"]) <= {
        "SL",
        "BE",
        "TSL",
        "Target",
        "Score",
        "Time",
        "Replaced",
        "Open",
        "Delisted",
    }


def test_no_look_ahead() -> None:
    p, s, g = _setup()
    a = _run(p, s, g)
    cut = p.close.index[200]
    early = (p.close.index < cut)[:, None]
    q = replace(
        p,
        **{
            k: getattr(p, k).where(np.broadcast_to(early, p.close.shape), getattr(p, k) * 1.5)
            for k in ("open", "high", "low", "close")
        },
    )
    b = _run(q, s, g)
    fa = a.fills[pd.to_datetime(a.fills["date"]) < cut].reset_index(drop=True)
    fb = b.fills[pd.to_datetime(b.fills["date"]) < cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(fa, fb)


def test_resume_day_by_day_equals_backtest() -> None:
    import json

    from quantswing.portfolio.swing import SwingSim, SwingState

    p, s, g = _setup()
    idx = p.close.index
    full = run_swing(p, s, g, CFG, start=idx[80])
    sim = SwingSim(p, s, g, CFG)
    st = sim.init_state()
    eq: list[float] = []
    fills: list[pd.DataFrame] = []
    cuts = [(80, 120), (121, 121), (122, 180), (181, len(idx) - 1)]
    for a, b in cuts:
        last = b == len(idx) - 1
        st, r = sim.step(st, a, b, plan_last=not last)
        st = SwingState.from_json(json.loads(json.dumps(st.to_json())))
        eq += r.daily["equity"].tolist()
        fills.append(r.fills)
    assert np.allclose(eq, full.daily["equity"].to_numpy(), atol=1e-6)
    fa = pd.concat(fills, ignore_index=True)
    assert len(fa) == len(full.fills) and np.allclose(fa["px"], full.fills["px"])
