"""Order state machine (04 §7).

::

    PENDING(next open) → FILLED                       entries / adds (market-on-open)
    PENDING → EXPIRED                                 not executed (below min trade, no cash)
    EXIT_PENDING → CLOSED                             exits / trims
    EXIT_PENDING → EXIT_BLOCKED → EXIT_PENDING …      lower circuit: retried next session
    any open state → CANCELLED                        superseded by a newer plan / operator

``TRIGGERED`` is reserved for intraday stop-entry sleeves (not used by the portfolio).
Every transition is written to ``order_log`` with timestamp, price, reason, model_version.
"""

from __future__ import annotations

from enum import StrEnum


class OrderState(StrEnum):
    """Order states."""

    PENDING = "PENDING"
    TRIGGERED = "TRIGGERED"
    FILLED = "FILLED"
    EXPIRED = "EXPIRED"
    EXIT_PENDING = "EXIT_PENDING"
    EXIT_BLOCKED = "EXIT_BLOCKED"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


S = OrderState
ALLOWED: dict[OrderState, frozenset[OrderState]] = {
    S.PENDING: frozenset({S.TRIGGERED, S.FILLED, S.EXPIRED, S.CANCELLED}),
    S.TRIGGERED: frozenset({S.FILLED, S.EXPIRED, S.CANCELLED}),
    S.EXIT_PENDING: frozenset({S.CLOSED, S.EXIT_BLOCKED, S.EXPIRED, S.CANCELLED}),
    S.EXIT_BLOCKED: frozenset({S.EXIT_PENDING, S.CLOSED, S.CANCELLED, S.EXPIRED}),
    S.FILLED: frozenset(),
    S.CLOSED: frozenset(),
    S.EXPIRED: frozenset(),
    S.CANCELLED: frozenset(),
}
OPEN_STATES = frozenset({S.PENDING, S.TRIGGERED, S.EXIT_PENDING, S.EXIT_BLOCKED})


class IllegalTransition(ValueError):
    """A state change the machine does not allow."""


def check(src: OrderState, dst: OrderState) -> None:
    """Raise :class:`IllegalTransition` unless ``src → dst`` is allowed."""
    if dst not in ALLOWED[src]:
        raise IllegalTransition(f"{src} -> {dst}")
