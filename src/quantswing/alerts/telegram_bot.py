"""Telegram alerts, commands and the daily-login relay (D17, 06 §3).

Only messages from ``TELEGRAM_CHAT_ID`` are read. The bot token lives in ``.env`` and is
never logged (request URLs contain it, so URLs are never logged either).

Commands: ``/status /positions /pause /resume /flatten yes /report /help``. Controls are
written to the paper book and applied at the next daily run's close decision.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import requests

from quantswing.core.logging import get_logger
from quantswing.execution.paper_broker import PaperBook

log = get_logger(__name__)
API = "https://api.telegram.org/bot{token}/{method}"
MAX_LEN = 3900

HELP = (
    "/status - equity, regime, positions\n"
    "/positions - holdings\n"
    "/pause - no new stocks bought (holdings kept)\n"
    "/resume - undo /pause and /flatten\n"
    "/flatten yes - sell everything at the next open\n"
    "/report - last daily report\n"
    "Daily login: when asked, open the link, log in, then paste the full address of the "
    "page you land on here."
)


class Telegram:
    """Minimal Bot API client (requests; corporate SSL handled by pip-system-certs)."""

    def __init__(
        self, token: str | None, chat_id: str | None, timeout: float = 20, http: Any = None
    ) -> None:
        self.token, self.chat_id, self.timeout = token or "", str(chat_id or ""), timeout
        self.http = http or requests

    @classmethod
    def from_env(cls, timeout: float = 20) -> Telegram:
        """Client from ``.env`` (disabled when the keys are missing)."""
        from quantswing.core.config import get_secrets

        try:
            s = get_secrets()
        except Exception:
            return cls(None, None, timeout)
        tok = s.TELEGRAM_BOT_TOKEN.get_secret_value() if s.TELEGRAM_BOT_TOKEN else None
        return cls(tok, s.TELEGRAM_CHAT_ID, timeout)

    @property
    def enabled(self) -> bool:
        """True when both token and chat id are configured."""
        return bool(self.token and self.chat_id)

    def call(self, method: str, params: dict[str, Any]) -> Any:
        """Bot API call → ``result`` (``None`` on any failure; never raises)."""
        if not self.token:
            return None
        try:
            r = self.http.post(
                API.format(token=self.token, method=method), json=params, timeout=self.timeout
            )
            body = r.json()
        except Exception as e:
            log.warning("telegram call failed", extra={"method": method, "error": type(e).__name__})
            return None
        if not body.get("ok"):
            log.warning("telegram error", extra={"method": method, "desc": body.get("description")})
            return None
        return body.get("result")

    def send(self, text: str) -> bool:
        """Send plain text (split into chunks). True if every chunk was accepted."""
        if not self.enabled:
            return False
        ok = True
        for k in range(0, max(len(text), 1), MAX_LEN):
            res = self.call(
                "sendMessage",
                {
                    "chat_id": self.chat_id,
                    "text": text[k : k + MAX_LEN],
                    "disable_web_page_preview": True,
                },
            )
            ok = ok and res is not None
        return ok

    def updates(self, offset: int | None, wait_s: int = 0) -> list[dict[str, Any]]:
        """New updates after ``offset``."""
        params: dict[str, Any] = {"timeout": wait_s, "allowed_updates": ["message"]}
        if offset is not None:
            params["offset"] = offset
        res = self.call("getUpdates", params)
        return list(res) if isinstance(res, list) else []

    def owner_texts(self, ups: list[dict[str, Any]]) -> list[str]:
        """Texts sent by the configured chat only."""
        out = []
        for u in ups:
            m = u.get("message") or {}
            if str((m.get("chat") or {}).get("id")) == self.chat_id and m.get("text"):
                out.append(str(m["text"]).strip())
        return out


def _next_offset(ups: list[dict[str, Any]], cur: int | None) -> int | None:
    ids = [int(u["update_id"]) for u in ups if "update_id" in u]
    return max(ids) + 1 if ids else cur


def handle(text: str, book: PaperBook) -> str:
    """Reply to one command (controls are persisted in ``book``)."""
    cmd = text.split()[0].lower().split("@")[0] if text else ""
    arg = text.split()[1].lower() if len(text.split()) > 1 else ""
    if cmd == "/pause":
        book.set_control("pause_entries", True)
        return "Paused: no new stocks will be bought. Holdings are kept. /resume to undo."
    if cmd == "/resume":
        book.set_control("pause_entries", False)
        book.set_control("flatten", False)
        return "Resumed: normal trading from the next plan."
    if cmd == "/flatten":
        if arg != "yes":
            return "This sells EVERYTHING at the next open. Send: /flatten yes"
        book.set_control("flatten", True)
        return "Flatten armed: everything is sold at the next open after tonight's run."
    if cmd == "/status":
        return status_text(book)
    if cmd == "/positions":
        pos = book.positions()
        if pos.empty:
            return "No positions."
        lines = [
            f"{str(r['symbol']).split(':')[-1]:<14} {float(r['qty']):>7.0f} @ "
            f"{float(r['last_px']):>9.2f}  {float(r['weight']):>6.1%}"
            for r in pos.to_dict("records")
        ]
        return "Positions\n" + "\n".join(lines)
    if cmd == "/report":
        return str(book.get("last_report", "No report yet."))
    return HELP


def status_text(book: PaperBook) -> str:
    """One-screen account status."""
    eq = book.equity()
    if eq.empty:
        return "Paper account not started yet."
    last = eq.iloc[-1]
    start = float(book.get("start_capital", eq["equity"].iloc[0]))
    ctl = []
    if book.control("pause_entries"):
        ctl.append("entries PAUSED")
    if book.control("flatten"):
        ctl.append("FLATTEN armed")
    return (
        f"Paper {eq.index[-1]:%d-%b-%Y}\n"
        f"Equity Rs {last['equity']:,.0f} ({last['equity'] / start - 1:+.2%} since start)\n"
        f"Drawdown {last['dd']:.1%} | Positions {int(last['positions'])} | "
        f"Invested {last['invested'] / last['equity']:.0%}\n"
        f"Regime {last['regime'] or '-'}" + (f"\nControls: {', '.join(ctl)}" if ctl else "")
    )


def process_commands(tg: Telegram, book: PaperBook) -> int:
    """Answer pending owner commands. Returns how many were handled."""
    if not tg.enabled:
        return 0
    off = book.get("tg_offset")
    ups = tg.updates(off)
    texts = [t for t in tg.owner_texts(ups) if t.startswith("/")]
    for t in texts:
        tg.send(handle(t, book))
    book.put("tg_offset", _next_offset(ups, off))
    return len(texts)


def wait_for_auth(
    tg: Telegram,
    book: PaperBook,
    login_url: str,
    wait_minutes: int,
    poll_seconds: int,
    login: Callable[[str], Any],
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Send the Fyers login link; wait for the redirect URL; exchange it via ``login``."""
    if not tg.enabled:
        return False
    off = _next_offset(tg.updates(book.get("tg_offset")), book.get("tg_offset"))
    tg.send(
        "Fyers login needed for today's paper run.\n"
        f"1) Open: {login_url}\n"
        "2) Log in.\n"
        "3) Copy the FULL address of the page you land on (it may say 'can't connect') "
        "and paste it here."
    )
    deadline = time.monotonic() + wait_minutes * 60
    while time.monotonic() < deadline:
        ups = tg.updates(off, wait_s=min(poll_seconds, 50))
        off = _next_offset(ups, off)
        for t in tg.owner_texts(ups):
            if "auth_code" in t or (len(t) > 40 and " " not in t and not t.startswith("/")):
                try:
                    login(t)
                except Exception as e:
                    tg.send(f"Login failed ({type(e).__name__}). Please try the link again.")
                    continue
                book.put("tg_offset", off)
                tg.send("Login OK - running today's update.")
                return True
            if t.startswith("/"):
                tg.send(handle(t, book))
        book.put("tg_offset", off)
        if not ups:
            sleep(poll_seconds)
    tg.send("No login received - today's paper run is skipped. It catches up next time.")
    return False
