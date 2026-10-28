"""Fyers v3 authentication, implemented exactly per 09_FYERS_AUTH_REFERENCE.

Flow (09 §1)
------------
1. Login URL  — ``fyersModel.SessionModel(client_id, redirect_uri, response_type="code",
   state=...)`` → ``generate_authcode()``.
2. User logs in in the browser; the redirect URL carries ``auth_code=...``.
3. Capture    — accept the full redirect URL or the raw code; take ``auth_code=`` up to ``&``.
4. Exchange   — DIRECT ``POST https://api-t1.fyers.in/api/v3/validate-authcode`` with JSON
   ``{grant_type: "authorization_code", appIdHash: sha256("<client_id>:<secret>"), code}``,
   timeout 15 s. (The SDK's ``generate_token()`` calls ``.json()`` without checking the HTTP
   status and crashes with ``Expecting value: line 1 column 1`` on an empty non-200 body.)
5. Validate   — empty body → "auth_code used/expired"; non-JSON → status + body;
   ``s != "ok"`` or no ``access_token`` → raise with the response.
6. Store      — Fernet-encrypted at rest (06 §5); valid for the IST day it was obtained.
7. Client     — ``fyersModel.FyersModel(client_id, token=access_token, is_async=False,
   log_path="")``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from cryptography.fernet import Fernet, InvalidToken
from fyers_apiv3 import fyersModel

from quantswing.core.clock import IST, now_ist
from quantswing.core.config import AppConfig, Secrets, get_config, get_secrets
from quantswing.core.logging import add_secret, get_logger
from quantswing.core.types import AuthError, TokenMissingError

log = get_logger(__name__)

AUTH_CODE_KEY = "auth_code="


# --------------------------------------------------------------------------- steps 1–3
def build_login_url(client_id: str, redirect_uri: str, cfg: AppConfig | None = None) -> str:
    """Step 1: Fyers login URL via ``SessionModel.generate_authcode()``."""
    fc = (cfg or get_config()).settings.fyers
    session = fyersModel.SessionModel(
        client_id=client_id,
        redirect_uri=redirect_uri,
        response_type=fc.response_type,
        state=fc.login_url_state,
    )
    return str(session.generate_authcode())


def parse_auth_code(text: str) -> str:
    """Step 3: extract the auth code from a full redirect URL or a raw code.

    Rule (09 §1): if ``auth_code=`` is present, take everything after it up to the next
    ``&`` (or end); otherwise the stripped input *is* the code.

    >>> parse_auth_code("https://x/cb?s=ok&code=200&auth_code=abc.def&state=q")
    'abc.def'
    >>> parse_auth_code("  abc.def  ")
    'abc.def'
    """
    s = text.strip().strip('"').strip("'")
    if AUTH_CODE_KEY in s:
        s = s.split(AUTH_CODE_KEY, 1)[1].split("&", 1)[0]
    s = s.strip().rstrip("#")
    if not s:
        raise AuthError("No auth_code found in input (paste the full redirect URL or the code)")
    if any(c.isspace() for c in s):
        raise AuthError("auth_code contains whitespace — paste the URL/code exactly")
    return s


# --------------------------------------------------------------------------- steps 4–5
def app_id_hash(client_id: str, secret_key: str) -> str:
    """``sha256("<client_id>:<secret_key>")`` hex digest (09 §1 step 4)."""
    return hashlib.sha256(f"{client_id}:{secret_key}".encode()).hexdigest()


def exchange_auth_code(
    client_id: str,
    secret_key: str,
    auth_code: str,
    *,
    cfg: AppConfig | None = None,
    http: Any = requests,
) -> str:
    """Steps 4–5: direct POST to validate-authcode; return the access token.

    ``http`` is injectable (anything with ``.post(url, json=, timeout=)``) for tests.
    Raises :class:`AuthError` with a clear message on every failure mode.
    """
    fc = (cfg or get_config()).settings.fyers
    payload = {
        "grant_type": fc.grant_type,
        "appIdHash": app_id_hash(client_id, secret_key),
        "code": auth_code,
    }
    try:
        resp = http.post(fc.validate_authcode_url, json=payload, timeout=fc.request_timeout_s)
    except requests.RequestException as e:
        raise AuthError(f"validate-authcode request failed: {type(e).__name__}: {e}") from e

    status = getattr(resp, "status_code", None)
    body = (getattr(resp, "text", "") or "").strip()
    if not body:
        raise AuthError(
            f"validate-authcode returned an empty body (HTTP {status}): "
            "auth_code used/expired — log in again and paste a fresh redirect URL"
        )
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        raise AuthError(f"validate-authcode non-JSON response (HTTP {status}): {body[:500]}") from e

    if (
        not isinstance(data, dict)
        or data.get("s") != fc.success_status
        or not data.get("access_token")
    ):
        raise AuthError(f"validate-authcode rejected (HTTP {status})", response=data)
    token = str(data["access_token"])
    add_secret(token)
    log.info("access token obtained", extra={"http_status": status})
    return token


# --------------------------------------------------------------------------- step 6
@dataclass(frozen=True)
class StoredToken:
    """Decrypted token record."""

    client_id: str
    access_token: str
    obtained_at: datetime  # tz-aware IST

    def is_valid_today(self, now: datetime | None = None) -> bool:
        """True if obtained on the current IST date (Fyers tokens expire daily)."""
        now = now or now_ist()
        return self.obtained_at.astimezone(IST).date() == now.astimezone(IST).date()


class TokenStore:
    """Fernet-encrypted token file. Key lives in a separate file (both git-ignored).

    On first use a new key is generated. On Windows, keep the repo inside your user
    profile (NTFS permissions) and/or enable BitLocker — the key file is the only thing
    protecting the token at rest.
    """

    def __init__(self, token_file: Path, key_file: Path) -> None:
        self.token_file = token_file
        self.key_file = key_file

    @classmethod
    def from_config(cls, cfg: AppConfig | None = None) -> TokenStore:
        """Build from ``settings.paths``."""
        p = (cfg or get_config()).settings.paths
        return cls(p.resolve(p.token_file), p.resolve(p.token_key_file))

    def _fernet(self) -> Fernet:
        if not self.key_file.exists():
            self.key_file.parent.mkdir(parents=True, exist_ok=True)
            self.key_file.write_bytes(Fernet.generate_key())
            _restrict(self.key_file)
        return Fernet(self.key_file.read_bytes().strip())

    def save(self, client_id: str, access_token: str, obtained_at: datetime | None = None) -> None:
        """Encrypt and write the token record."""
        rec = {
            "client_id": client_id,
            "access_token": access_token,
            "obtained_at": (obtained_at or now_ist()).isoformat(),
        }
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        self.token_file.write_bytes(self._fernet().encrypt(json.dumps(rec).encode()))
        _restrict(self.token_file)

    def load(self) -> StoredToken | None:
        """Decrypt the token record, or ``None`` if absent/unreadable."""
        if not self.token_file.exists() or not self.key_file.exists():
            return None
        try:
            raw = self._fernet().decrypt(self.token_file.read_bytes())
            d = json.loads(raw)
            tok = StoredToken(
                d["client_id"], d["access_token"], datetime.fromisoformat(d["obtained_at"])
            )
        except (InvalidToken, ValueError, KeyError) as e:
            log.warning("stored token unreadable", extra={"error": type(e).__name__})
            return None
        add_secret(tok.access_token)
        return tok

    def clear(self) -> None:
        """Delete the stored token (not the key)."""
        self.token_file.unlink(missing_ok=True)


def _restrict(path: Path) -> None:
    """Best-effort owner-only permissions (no-op semantics on Windows ACLs)."""
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)


# --------------------------------------------------------------------------- step 7
def make_client(client_id: str, access_token: str) -> fyersModel.FyersModel:
    """Step 7: authenticated SDK client."""
    return fyersModel.FyersModel(
        client_id=client_id, token=access_token, is_async=False, log_path=""
    )


def get_valid_token(
    store: TokenStore | None = None, secrets: Secrets | None = None, cfg: AppConfig | None = None
) -> StoredToken:
    """Return today's stored token for the configured app, else raise TokenMissingError."""
    cfg = cfg or get_config()
    store = store or TokenStore.from_config(cfg)
    secrets = secrets or get_secrets()
    tok = store.load()
    cid = secrets.FYERS_CLIENT_ID.get_secret_value()
    if tok is None or tok.client_id != cid:
        raise TokenMissingError("No stored Fyers token — run: python scripts/fyers_login.py")
    if cfg.settings.fyers.token_valid_same_ist_day and not tok.is_valid_today():
        raise TokenMissingError(
            f"Fyers token from {tok.obtained_at.date()} has expired — "
            "run: python scripts/fyers_login.py"
        )
    return tok


def get_fyers_client(store: TokenStore | None = None) -> fyersModel.FyersModel:
    """Authenticated client from today's stored token."""
    tok = get_valid_token(store)
    return make_client(tok.client_id, tok.access_token)


def login_with_input(user_input: str, store: TokenStore | None = None) -> StoredToken:
    """Steps 3–6: parse pasted URL/code, exchange, store. Returns the stored record."""
    s = get_secrets()
    cid = s.FYERS_CLIENT_ID.get_secret_value()
    code = parse_auth_code(user_input)
    token = exchange_auth_code(cid, s.FYERS_SECRET_KEY.get_secret_value(), code)
    store = store or TokenStore.from_config()
    obtained = now_ist()
    store.save(cid, token, obtained)
    return StoredToken(cid, token, obtained)
