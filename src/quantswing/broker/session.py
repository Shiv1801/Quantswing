"""Convenience: authenticated, throttled :class:`FyersData` from today's stored token."""

from __future__ import annotations

from quantswing.broker.fyers_auth import TokenStore, get_valid_token, make_client
from quantswing.broker.fyers_data import FyersData
from quantswing.broker.throttler import Throttler
from quantswing.core.config import AppConfig, get_config


def open_data_client(cfg: AppConfig | None = None) -> FyersData:
    """Return a ready :class:`FyersData`. Raises ``TokenMissingError`` if not logged in today."""
    cfg = cfg or get_config()
    tok = get_valid_token(TokenStore.from_config(cfg), cfg=cfg)
    return FyersData(make_client(tok.client_id, tok.access_token), Throttler.from_config(cfg), cfg)
