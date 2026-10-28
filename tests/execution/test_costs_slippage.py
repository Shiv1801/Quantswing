"""Cost model against hand-computed Fyers delivery charges; slippage formula."""

from __future__ import annotations

import pytest

from quantswing.core.config import get_config
from quantswing.execution.costs import CostModel
from quantswing.execution.slippage import SlippageModel

C = get_config().costs


def test_buy_one_lakh() -> None:
    b = CostModel(C).fill("buy", 100, 1000.0)
    assert b.brokerage == pytest.approx(20.0)  # min(20, 0.3% of 1,00,000)
    assert b.stt == pytest.approx(100.0)
    assert b.txn == pytest.approx(3.0699)
    assert b.sebi == pytest.approx(0.1)
    assert b.stamp == pytest.approx(15.0)
    assert b.dp == 0.0
    assert b.gst == pytest.approx(0.18 * (20 + 3.0699 + 0.1))
    assert b.total == pytest.approx(142.3405, abs=1e-3)


def test_sell_one_lakh_with_dp_once_per_day() -> None:
    m = CostModel(C)
    s1 = m.fill("sell", 100, 1000.0)
    assert s1.stamp == 0.0 and s1.dp == pytest.approx(12.5 * 1.18)
    assert s1.total == pytest.approx(20 + 100 + 3.0699 + 0.1 + 14.75 + 0.18 * 23.1699, abs=1e-3)
    assert m.fill("sell", 100, 1000.0, first_sell_today=False).dp == 0.0


def test_small_order_percent_brokerage_and_stress() -> None:
    assert CostModel(C).fill("buy", 10, 100.0).brokerage == pytest.approx(3.0)
    assert CostModel(C, 2.0).fill("buy", 100, 1000.0).total == pytest.approx(2 * 142.3405, abs=2e-3)


def test_slippage() -> None:
    s = SlippageModel(C.slippage)
    assert s.frac(1000, 100_000, 0.02) == pytest.approx(0.0005 + 0.1 * 0.02 * 0.1)
    assert s.price("buy", 100.0, 1000, 100_000, 0.02) == pytest.approx(100.07)
    assert s.price("sell", 100.0, 1000, 100_000, 0.02) == pytest.approx(99.93)
    assert s.frac(10, 0, float("nan")) == pytest.approx(0.0005)
