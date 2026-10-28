"""One-time Telegram setup: saves TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID into .env.

Before running: in Telegram, open @BotFather, send /newbot, follow the steps and copy the
token it gives you. Then run this script, paste the token (it is not shown), and send any
message (e.g. "hi") to your new bot when asked.

Usage::

    python scripts/telegram_setup.py
"""

from __future__ import annotations

import getpass
import sys
import time

from quantswing.alerts.telegram_bot import Telegram
from quantswing.core.config import REPO_ROOT


def set_env(key: str, value: str) -> None:
    """Add or replace ``key=value`` in .env (other lines untouched)."""
    env = REPO_ROOT / ".env"
    lines = env.read_text(encoding="utf-8").splitlines() if env.exists() else []
    out = [ln for ln in lines if not ln.strip().startswith(f"{key}=")]
    out.append(f"{key}={value}")
    env.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> int:
    """Entry point."""
    token = getpass.getpass("Paste the bot token from @BotFather (hidden): ").strip()
    tg = Telegram(token, "0")
    me = tg.call("getMe", {})
    if not me:
        print("[XX] Telegram did not accept that token. Check it and try again.")
        return 1
    print(f"[OK] Bot @{me.get('username')} found.")
    print(f"Now open Telegram, find @{me.get('username')}, press Start and send: hi")
    off = None
    for _ in range(60):
        ups = tg.updates(off, wait_s=5)
        for u in ups:
            off = int(u["update_id"]) + 1
            chat = (u.get("message") or {}).get("chat") or {}
            if chat.get("id"):
                cid = str(chat["id"])
                set_env("TELEGRAM_BOT_TOKEN", token)
                set_env("TELEGRAM_CHAT_ID", cid)
                Telegram(token, cid).send("QuantSwing is connected. Send /help for commands.")
                print("[OK] Saved to .env. You should get a message from the bot now.")
                return 0
        time.sleep(1)
    print("[XX] No message received within ~5 minutes. Run the script again.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
