"""Structured JSON logging with secret masking (06 §5, 08 §3).

Every record is one JSON object per line::

    {"ts": "2026-09-25T09:15:00.123+05:30", "level": "INFO", "logger": "...",
     "msg": "...", "model_version": "...", ...extra fields}

Masking (applied to the message, args and every ``extra`` field):
* dict keys / ``key=value`` / ``"key": "value"`` pairs whose key is in
  ``settings.logging.mask_keys`` → value replaced by ``abcd****``;
* literal values of env vars listed in ``settings.logging.mask_env_vars`` → masked anywhere.

Usage::

    from quantswing.core.logging import setup_logging, get_logger
    setup_logging()
    log = get_logger(__name__)
    log.info("history fetched", extra={"symbol": "NSE:RELIANCE-EQ", "rows": 248})
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections.abc import Iterable, Mapping
from logging.handlers import RotatingFileHandler
from typing import Any

from quantswing.core.clock import now_ist
from quantswing.core.config import AppConfig, get_config

_STD_ATTRS: frozenset[str] = frozenset(
    vars(logging.LogRecord("x", 0, "x", 0, "x", None, None)).keys()
) | {"message", "asctime"}

_MASK = "****"


class SecretMasker:
    """Masks secrets in strings and nested structures.

    Inputs: ``keys`` — sensitive key names (case-insensitive); ``literals`` — exact secret
    values to scrub anywhere; ``visible`` — leading chars kept visible.
    """

    def __init__(self, keys: Iterable[str], literals: Iterable[str], visible: int) -> None:
        self.keys = {k.lower() for k in keys}
        self.literals = sorted({s for s in literals if s}, key=len, reverse=True)
        self.visible = visible
        alt = "|".join(re.escape(k) for k in sorted(self.keys, key=len, reverse=True))
        # key=value (query strings / kv text) and "key": "value" (JSON-ish text)
        self._kv = re.compile(rf"(?i)\b({alt})=([^&\s,;\"']+)") if alt else None
        self._json = re.compile(rf"(?i)([\"']({alt})[\"']\s*:\s*[\"'])([^\"']+)") if alt else None

    def mask_value(self, value: str) -> str:
        """Return ``value`` with all but the first ``visible`` chars hidden."""
        return value[: self.visible] + _MASK if len(value) > self.visible else _MASK

    def mask_str(self, text: str) -> str:
        """Mask secret literals and sensitive key/value pairs inside free text."""
        for lit in self.literals:
            if lit in text:
                text = text.replace(lit, self.mask_value(lit))
        if self._kv is not None:
            text = self._kv.sub(lambda m: f"{m.group(1)}={self.mask_value(m.group(2))}", text)
        if self._json is not None:
            text = self._json.sub(lambda m: m.group(1) + self.mask_value(m.group(3)), text)
        return text

    def mask(self, obj: Any, key: str | None = None) -> Any:
        """Recursively mask ``obj``; values under sensitive keys are always masked."""
        if key is not None and key.lower() in self.keys and obj is not None:
            return self.mask_value(str(obj))
        if isinstance(obj, str):
            return self.mask_str(obj)
        if isinstance(obj, Mapping):
            return {k: self.mask(v, str(k)) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self.mask(v) for v in obj]
        return obj


class JsonFormatter(logging.Formatter):
    """Formats records as single-line JSON with IST timestamps and masking."""

    def __init__(self, masker: SecretMasker, model_version: str) -> None:
        super().__init__()
        self.masker = masker
        self.model_version = model_version

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` to a JSON string."""
        payload: dict[str, Any] = {
            "ts": now_ist().isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": self.masker.mask_str(record.getMessage()),
            "model_version": self.model_version,
        }
        for k, v in record.__dict__.items():
            if k not in _STD_ATTRS and not k.startswith("_"):
                payload[k] = self.masker.mask(v, k)
        if record.exc_info:
            payload["exc"] = self.masker.mask_str(self.formatException(record.exc_info))
        return json.dumps(payload, default=str, ensure_ascii=True)


def build_masker(cfg: AppConfig) -> SecretMasker:
    """Create a masker from config + current secret env values."""
    lc = cfg.settings.logging
    literals = [os.environ.get(name, "") for name in lc.mask_env_vars]
    try:  # also pick up values that only live in .env
        from quantswing.core.config import get_secrets

        s = get_secrets()
        literals += [s.FYERS_CLIENT_ID.get_secret_value(), s.FYERS_SECRET_KEY.get_secret_value()]
        if s.TELEGRAM_BOT_TOKEN is not None:
            literals.append(s.TELEGRAM_BOT_TOKEN.get_secret_value())
    except Exception:
        pass
    return SecretMasker(lc.mask_keys, literals, lc.mask_visible_chars)


_configured = False
_masker: SecretMasker | None = None


def setup_logging(cfg: AppConfig | None = None, *, force: bool = False) -> None:
    """Configure root logging: rotating JSON file in ``paths.log_dir`` + optional console."""
    global _configured, _masker
    if _configured and not force:
        return
    cfg = cfg or get_config()
    lc = cfg.settings.logging
    log_dir = cfg.settings.paths.resolve(cfg.settings.paths.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    _masker = build_masker(cfg)
    fmt = JsonFormatter(_masker, cfg.settings.app.model_version)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(lc.level)

    fh = RotatingFileHandler(
        log_dir / lc.file_name, maxBytes=lc.max_bytes, backupCount=lc.backup_count, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)
    if lc.console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(fmt)
        ch.setLevel(lc.console_level)
        root.addHandler(ch)
    _configured = True


def add_secret(value: str) -> None:
    """Register a runtime secret (e.g. a fresh access token) for masking."""
    if _masker is not None and value and value not in _masker.literals:
        _masker.literals = sorted([*_masker.literals, value], key=len, reverse=True)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (call :func:`setup_logging` once at program start)."""
    return logging.getLogger(name)
