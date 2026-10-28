"""Equity-delivery transaction cost model (04 §2) from ``config/costs.yaml``.

Per fill (value ``V = qty × price``):

| Component | Buy | Sell |
|---|---|---|
| Brokerage | ``min(flat, pct·V)`` | same |
| STT | ``stt·V`` | ``stt·V`` |
| NSE transaction | ``txn·V`` | ``txn·V`` |
| SEBI fee | ``V/1e7 · fee_per_crore`` | same |
| Stamp duty | ``stamp·V`` | — |
| DP charge | — | ``dp`` per scrip per sell day (× (1+GST) if ``gst_applies``) |
| GST | ``gst · (brokerage + txn + SEBI)`` | same |

``multiplier`` scales every component (05 §3 cost-stress run).
"""

from __future__ import annotations

from dataclasses import dataclass

from quantswing.core.config import CostsCfg

CRORE = 1e7


@dataclass(frozen=True)
class CostBreakdown:
    """Charges for one fill (₹)."""

    brokerage: float
    stt: float
    txn: float
    sebi: float
    stamp: float
    dp: float
    gst: float

    @property
    def total(self) -> float:
        """Sum of all charges."""
        return self.brokerage + self.stt + self.txn + self.sebi + self.stamp + self.dp + self.gst


class CostModel:
    """Computes delivery charges for buy and sell fills."""

    def __init__(self, cfg: CostsCfg, multiplier: float = 1.0) -> None:
        self.c = cfg
        self.m = multiplier

    def fill(
        self, side: str, qty: float, price: float, *, first_sell_today: bool = True
    ) -> CostBreakdown:
        """Charges for a ``side`` (``"buy"``/``"sell"``) fill of ``qty`` at ``price``.

        ``first_sell_today``: DP charge applies once per scrip per sell day.
        """
        if side not in ("buy", "sell"):
            raise ValueError(side)
        v = abs(qty) * price
        b, st = self.c.broker, self.c.statutory
        brokerage = (
            min(b.delivery_brokerage_flat_inr, b.delivery_brokerage_pct * v) if v > 0 else 0.0
        )
        stt = st.stt_delivery_pct * v
        txn = st.nse_txn_pct * v
        sebi = v / CRORE * st.sebi_fee_per_crore_inr
        stamp = st.stamp_duty_buy_pct * v if side == "buy" else 0.0
        dp = 0.0
        if side == "sell" and first_sell_today and v > 0:
            dp = self.c.dp_charge.per_scrip_per_sell_day_inr
            if self.c.dp_charge.gst_applies:
                dp *= 1 + st.gst_pct
        gst = st.gst_pct * (brokerage + txn + sebi)
        m = self.m
        return CostBreakdown(brokerage * m, stt * m, txn * m, sebi * m, stamp * m, dp * m, gst * m)
