"""IC and IC-IR combination, incl. point-in-time weights."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantswing.alpha.combine import composite, ic_ir_weights
from quantswing.alpha.ic import forward_log_return, rank_ic, summarise
from quantswing.core.config import get_config

CC = get_config().factors.ic.combine.model_copy(update={"sign_aware": False})  # spec rule


def test_rank_ic_perfect_and_inverse() -> None:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2021-01-01", periods=10)
    fwd = pd.DataFrame(rng.normal(size=(10, 50)), index=idx)
    assert rank_ic(fwd, fwd, 30).round(12).eq(1.0).all()
    assert rank_ic(-fwd, fwd, 30).round(12).eq(-1.0).all()
    assert rank_ic(fwd.iloc[:, :20], fwd.iloc[:, :20], 30).isna().all()


def test_forward_return_is_future() -> None:
    lp = pd.DataFrame({"A": [0.0, 0.1, 0.3]})
    assert list(forward_log_return(lp, 1)["A"].round(6).fillna(9)) == [0.1, 0.2, 9]


def test_summarise_nonoverlap_t() -> None:
    s = pd.Series(np.r_[np.full(200, 0.05)] + np.random.default_rng(1).normal(0, 0.02, 200))
    st = summarise(s, 20)
    assert st.n == 200 and st.mean == pytest.approx(0.05, abs=0.01)
    sub = s.iloc[::20]
    assert st.t_nonoverlap == pytest.approx(sub.mean() / sub.std() * np.sqrt(len(sub)))


def test_ic_ir_weights_prefer_positive_and_are_point_in_time() -> None:
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2019-01-01", periods=900)
    ic = pd.DataFrame(
        {"GOOD": 0.05 + rng.normal(0, 0.05, 900), "BAD": -0.03 + rng.normal(0, 0.05, 900)},
        index=idx,
    )
    w = ic_ir_weights(ic, 20, CC)
    last = w.dropna().iloc[-1]
    assert last["GOOD"] > last["BAD"] and last.sum() == pytest.approx(1.0)
    assert last["BAD"] == pytest.approx(CC.shrink_to_equal / 2)  # only the shrinkage share
    cut = 600
    ic2 = ic.copy()
    ic2.iloc[cut:] = -ic2.iloc[cut:]  # rewrite the future
    w2 = ic_ir_weights(ic2, 20, CC)
    upto = idx[cut + 20]  # IC(cut) only becomes known at cut+20
    pd.testing.assert_frame_equal(w.loc[:upto], w2.loc[:upto])


def test_composite_renormalises() -> None:
    idx = pd.bdate_range("2021-01-01", periods=2)
    a = pd.DataFrame({"X": [1.0, 1.0], "Y": [np.nan, 2.0]}, index=idx)
    b = pd.DataFrame({"X": [3.0, 3.0], "Y": [4.0, 4.0]}, index=idx)
    w = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]}, index=idx)
    c = composite([("A", a), ("B", b)], w)
    assert c.at[idx[0], "X"] == 2.0 and c.at[idx[0], "Y"] == 4.0 and c.at[idx[1], "Y"] == 3.0


def test_sign_aware_weights_use_negative_factors() -> None:
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2019-01-01", periods=900)
    ic = pd.DataFrame(
        {"POS": 0.04 + rng.normal(0, 0.05, 900), "NEG": -0.06 + rng.normal(0, 0.05, 900)}, index=idx
    )
    w_spec = ic_ir_weights(ic, 20, CC).dropna().iloc[-1]
    w_sa = ic_ir_weights(ic, 20, CC.model_copy(update={"sign_aware": True})).dropna().iloc[-1]
    assert w_spec["NEG"] > 0 and w_spec["POS"] > w_spec["NEG"]  # spec: NEG only gets shrinkage
    assert w_sa["NEG"] < 0 and abs(w_sa["NEG"]) > w_sa["POS"] > 0  # inverted and weighted by |IC|/σ
    assert w_sa.abs().sum() == pytest.approx(1.0)
    idx2 = pd.bdate_range("2021-01-01", periods=1)
    z = pd.DataFrame({"X": [1.0]}, index=idx2)
    w = pd.DataFrame({"A": [0.5], "B": [-0.5]}, index=idx2)
    c = composite([("A", z), ("B", -z)], w)
    assert c.iloc[0, 0] == pytest.approx(1.0)  # both agree after inversion
