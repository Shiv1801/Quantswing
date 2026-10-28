"""Auth: auth-code parsing, appIdHash, validate-authcode error handling, token store."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from quantswing.broker.fyers_auth import (
    TokenStore,
    app_id_hash,
    exchange_auth_code,
    parse_auth_code,
)
from quantswing.core.clock import IST
from quantswing.core.types import AuthError

CODE = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJhcGkubG9naW4.abc-_XYZ"


@pytest.mark.parametrize(
    "text",
    [
        CODE,
        f"  {CODE}\n",
        f'"{CODE}"',
        f"https://127.0.0.1/?s=ok&code=200&auth_code={CODE}&state=quantswing",
        f"https://127.0.0.1/?auth_code={CODE}",
        f"https://127.0.0.1/?s=ok&auth_code={CODE}&state=quantswing#",
        f"http://localhost:8080/cb?state=x&auth_code={CODE}",
    ],
)
def test_parse_auth_code_variants(text: str) -> None:
    assert parse_auth_code(text) == CODE


@pytest.mark.parametrize("bad", ["", "   ", "https://127.0.0.1/?auth_code=&state=x", "ab cd"])
def test_parse_auth_code_rejects(bad: str) -> None:
    with pytest.raises(AuthError):
        parse_auth_code(bad)


def test_app_id_hash() -> None:
    assert app_id_hash("ABC-100", "sek") == hashlib.sha256(b"ABC-100:sek").hexdigest()


class FakeResp:
    def __init__(self, status: int, text: str) -> None:
        self.status_code = status
        self.text = text


class FakeHTTP:
    def __init__(self, resp: FakeResp) -> None:
        self.resp = resp
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, json: dict[str, Any], timeout: float) -> FakeResp:
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return self.resp


def test_exchange_success_posts_exact_payload() -> None:
    http = FakeHTTP(FakeResp(200, json.dumps({"s": "ok", "access_token": "TOK123"})))
    assert exchange_auth_code("ABC-100", "sek", "code1", http=http) == "TOK123"
    call = http.calls[0]
    assert call["url"] == "https://api-t1.fyers.in/api/v3/validate-authcode"
    assert call["timeout"] == 15
    assert call["json"] == {
        "grant_type": "authorization_code",
        "appIdHash": app_id_hash("ABC-100", "sek"),
        "code": "code1",
    }


@pytest.mark.parametrize(
    ("resp", "match"),
    [
        (FakeResp(400, ""), "used/expired"),
        (FakeResp(502, "<html>Bad gateway</html>"), "non-JSON"),
        (FakeResp(200, json.dumps({"s": "error", "code": -413, "message": "invalid"})), "rejected"),
        (FakeResp(200, json.dumps({"s": "ok"})), "rejected"),
    ],
)
def test_exchange_failure_modes(resp: FakeResp, match: str) -> None:
    with pytest.raises(AuthError, match=match):
        exchange_auth_code("ABC-100", "sek", "c", http=FakeHTTP(resp))


def test_token_store_roundtrip_encrypted(tmp_path: Path) -> None:
    store = TokenStore(tmp_path / "t.enc", tmp_path / "k.key")
    obtained = datetime(2026, 9, 25, 9, 1, tzinfo=IST)
    store.save("ABC-100", "SECRET_TOKEN_VALUE", obtained)
    assert b"SECRET_TOKEN_VALUE" not in (tmp_path / "t.enc").read_bytes()
    tok = store.load()
    assert tok is not None and tok.access_token == "SECRET_TOKEN_VALUE"
    assert tok.is_valid_today(obtained + timedelta(hours=6))
    assert not tok.is_valid_today(obtained + timedelta(days=1))


def test_token_store_wrong_key_returns_none(tmp_path: Path) -> None:
    store = TokenStore(tmp_path / "t.enc", tmp_path / "k.key")
    store.save("ABC-100", "X")
    (tmp_path / "k.key").unlink()
    TokenStore(tmp_path / "t.enc", tmp_path / "k.key")._fernet()  # new key generated
    assert store.load() is None
