"""Daily Fyers login (09 §1): open login URL -> paste redirect URL or code -> encrypted token.

Usage (from repo root, venv active)::

    python scripts/fyers_login.py            # opens browser, prompts for paste
    python scripts/fyers_login.py --no-browser
    python scripts/fyers_login.py --code "<full redirect URL or auth_code>"
    python scripts/fyers_login.py --check    # only report whether today's token is valid
"""

from __future__ import annotations

import argparse
import sys
import webbrowser

from quantswing.broker.fyers_auth import (
    TokenStore,
    build_login_url,
    get_valid_token,
    login_with_input,
    make_client,
)
from quantswing.core.config import get_secrets
from quantswing.core.logging import get_logger, setup_logging
from quantswing.core.types import AuthError, TokenMissingError

log = get_logger("scripts.fyers_login")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    ap = argparse.ArgumentParser(description="Fyers daily login for QuantSwing")
    ap.add_argument("--code", help="full redirect URL or raw auth_code (skips the prompt)")
    ap.add_argument("--no-browser", action="store_true", help="print the URL; don't open it")
    ap.add_argument("--check", action="store_true", help="only check today's stored token")
    ap.add_argument("--force", action="store_true", help="log in even if today's token is valid")
    args = ap.parse_args(argv)

    setup_logging()
    store = TokenStore.from_config()

    if not args.force:
        try:
            tok = get_valid_token(store)
            print(
                f"[OK] Valid token already stored (obtained {tok.obtained_at:%Y-%m-%d %H:%M} IST)."
            )
            return 0
        except TokenMissingError as e:
            if args.check:
                print(f"[MISSING] {e}")
                return 1

    s = get_secrets()
    user_input = args.code
    if not user_input:
        url = build_login_url(s.FYERS_CLIENT_ID.get_secret_value(), s.FYERS_REDIRECT_URI)
        print("\n1) Log in to Fyers with this URL:\n\n   " + url + "\n")
        if not args.no_browser:
            webbrowser.open(url)
        print("2) After login you are redirected. Copy the FULL redirect URL")
        print("   (or just the auth_code) and paste it below.\n")
        user_input = input("Redirect URL or auth_code: ")

    try:
        tok = login_with_input(user_input, store)
    except AuthError as e:
        print(f"\n[FAIL] {e}")
        if e.response is not None:
            print(f"       response: {e.response}")
        log.error("login failed", extra={"error": str(e)})
        return 2

    prof = make_client(tok.client_id, tok.access_token).get_profile()
    if isinstance(prof, dict) and prof.get("s") == "ok":
        name = (prof.get("data") or {}).get("name", "?")
        print(f"\n[OK] Logged in as {name}. Token encrypted at {store.token_file}")
        return 0
    print(f"\n[WARN] Token stored but profile check failed: {prof}")
    return 3


if __name__ == "__main__":
    sys.exit(main())
