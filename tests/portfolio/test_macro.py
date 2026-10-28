"""D39: macro drivers, spill-over factors, correlation risk, profit protection."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from quantswing.core.config import get_config
from quantswing.factors import macro
from quantswing.ingest.macro import find_drivers
from quantswing.portfolio.swing import avg_corr_spike, run_swing
from tests.factors.test_factors import synth_panels
from tests.portfolio.test_swing import CFG, _setup

BASE = get_config()


def _macro_panels() -> object:
    p = synth_panels(420, 30, seed=5)
    rng = np.random.default_rng(2)
    idx = p.close.index
    cl = p.close
    m = pd.DataFrame(
        {
            "N50": p.bench,
            "SMALL": p.bench * np.exp(np.cumsum(rng.normal(0, 0.004, len(idx)))),
            "VIX": 15 * np.exp(np.cumsum(rng.normal(0, 0.03, len(idx)))),
            "GOLD": 100 * np.exp(np.cumsum(rng.normal(0, 0.008, len(idx)))),
            "IT": cl.iloc[:, :10].mean(axis=1).ffill().bfill(),
            "PHARMA": cl.iloc[:, 10:20].mean(axis=1).ffill().bfill(),
        },
        index=idx,
    )
    return replace(p, macro=m)


def test_rolling_beta_recovers_slope() -> None:
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0, 0.01, 300), index=idx)
    y = pd.DataFrame({"a": 2 * x + rng.normal(0, 0.002, 300), "b": -x + rng.normal(0, 0.002, 300)})
    b = macro.rolling_beta(y, x, 250, 120).iloc[-1]
    assert abs(b["a"] - 2) < 0.1 and abs(b["b"] + 1) < 0.1


def test_macro_factors_exist_and_no_look_ahead() -> None:
    p = _macro_panels()
    mc = BASE.factors.macro.model_copy(update={"beta_window": 120, "min_obs": 60})
    f = macro.compute(p, mc)  # type: ignore[arg-type]
    assert {"MAC_SIZE", "MAC_VIX", "MAC_GOLD", "MAC_SECTOR"} <= set(f)
    cut = p.close.index[300]  # type: ignore[attr-defined]
    m2 = p.macro.copy()  # type: ignore[attr-defined]
    m2.loc[cut:] *= 1.7
    c2 = p.close.copy()  # type: ignore[attr-defined]
    c2.loc[cut:] *= 1.3
    g = macro.compute(replace(p, macro=m2, close=c2), mc)  # type: ignore[type-var]
    for k in f:
        a, b = f[k].loc[: p.close.index[299]], g[k].loc[: p.close.index[299]]  # type: ignore[attr-defined]
        pd.testing.assert_frame_equal(a, b)


def test_correlation_spike_detected() -> None:
    idx = pd.bdate_range("2020-01-01", periods=700)
    rng = np.random.default_rng(1)
    common = rng.normal(0, 0.01, 700)
    load = np.where(np.arange(700) >= 650, 3.0, 0.2)[:, None]
    r = pd.DataFrame(common[:, None] * load + rng.normal(0, 0.01, (700, 50)), index=idx)
    sp = avg_corr_spike(r, 20, 90, 500)
    assert sp[660:].mean() > 0.8 and sp[400:600].mean() < 0.25


def test_book_correlation_limit_changes_selection() -> None:
    p = synth_panels(300, 40, seed=4)
    rng = np.random.default_rng(3)
    s = pd.DataFrame(rng.normal(0, 1, p.close.shape), index=p.close.index, columns=p.close.columns)
    g = p.close.notna()
    base = _swing(entry_hurdle_mult=0.0)
    a = run_swing(p, s, g, base, start=p.close.index[80])
    b = run_swing(
        p, s, g, _swing(entry_hurdle_mult=0.0, corr_book_max=0.05), start=p.close.index[80]
    )
    assert b.daily["positions"].mean() < a.daily["positions"].mean()


def _swing(**kw: object) -> object:
    sw = CFG.risk.swing.model_copy(update=kw)
    return CFG.model_copy(update={"risk": CFG.risk.model_copy(update={"swing": sw})})


def _cfg_corr() -> object:
    sw = CFG.risk.swing.model_copy(update={"corr_book_max": 0.05})
    return CFG.model_copy(update={"risk": CFG.risk.model_copy(update={"swing": sw})})


def test_macro_protection_tightens_stops() -> None:
    p, s, g = _setup()
    b = p.bench.copy()
    r = np.array(np.log(b).diff().fillna(0), dtype=float)
    r[100::15] = -0.05  # sharp market falls
    p = replace(p, bench=pd.Series(1000 * np.exp(np.cumsum(r)), index=b.index))
    ones = pd.DataFrame(1.0, index=p.close.index, columns=p.close.columns)
    zero = ones * 0.0
    ex = macro.Exposures(ones, zero, zero)
    sw_on = CFG.risk.swing.model_copy(update={"macro_protect": True})
    on = CFG.model_copy(update={"risk": CFG.risk.model_copy(update={"swing": sw_on})})
    a = run_swing(p, s, g, CFG, start=p.close.index[80], macro=ex)
    c = run_swing(p, s, g, on, start=p.close.index[80], macro=ex)
    assert (a.trades["exit_reason"] == "MacroTSL").sum() == 0
    assert (c.trades["exit_reason"] == "MacroTSL").sum() > 0


def test_find_drivers_from_master() -> None:
    t = [
        "NSE:NIFTYIT-INDEX",
        "NSE:NIFTYPHARMA-INDEX",
        "NSE:NIFTY OIL AND GAS-INDEX",
        "NSE:FINNIFTY-INDEX",
        "NSE:NIFTYGS10YR-INDEX",
        "NSE:HDFCGOLD-EQ",
        "NSE:RELIANCE-EQ",
    ]
    d = find_drivers(t)
    assert d["IT"] == "NSE:NIFTYIT-INDEX" and d["FINSERV"] == "NSE:FINNIFTY-INDEX"
    assert d["GSEC"] == "NSE:NIFTYGS10YR-INDEX" and d["GOLD"] == "NSE:HDFCGOLD-EQ"
    assert "METAL" not in d
