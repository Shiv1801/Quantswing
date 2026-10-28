"""Shared types and column contracts."""

from __future__ import annotations

from typing import Final

#: Column order of a candle DataFrame produced by ``broker.fyers_data``.
CANDLE_COLUMNS: Final[tuple[str, ...]] = ("ts", "open", "high", "low", "close", "volume")

#: Fyers minute resolutions (history cap: ``intraday`` days/request).
INTRADAY_RESOLUTIONS: Final[frozenset[str]] = frozenset(
    {"1", "2", "3", "5", "10", "15", "20", "30", "45", "60", "120", "180", "240"}
)
#: Fyers daily resolutions.
DAILY_RESOLUTIONS: Final[frozenset[str]] = frozenset({"D", "1D"})
#: Fyers second resolutions.
SECOND_RESOLUTIONS: Final[frozenset[str]] = frozenset({"5S", "10S", "15S", "30S", "45S"})


class BrokerError(RuntimeError):
    """Error returned by the broker API (carries the raw response)."""

    def __init__(self, message: str, response: object | None = None) -> None:
        super().__init__(message)
        self.response = response


class AuthError(BrokerError):
    """Authentication / token-exchange failure."""


class TokenMissingError(AuthError):
    """No valid access token stored for today — run ``scripts/fyers_login.py``."""


class RateLimitExhausted(BrokerError):
    """Daily API quota exhausted (fatal for the rest of the day)."""
