"""D40 Login page: Fyers login from the app (one click with a local redirect URL, else paste)."""

from __future__ import annotations

import webbrowser

import streamlit as st

from quantswing.app.dashboard.ui import Ctx, muted
from quantswing.broker import callback


def token_status() -> tuple[bool, str]:
    """(logged in today?, text)."""
    from quantswing.broker.fyers_auth import TokenStore, get_valid_token
    from quantswing.core.types import TokenMissingError

    try:
        tok = get_valid_token(TokenStore.from_config())
    except TokenMissingError:
        return False, "Not logged in to Fyers today."
    except Exception:
        return (
            False,
            "Fyers keys are missing in .env (FYERS_CLIENT_ID, FYERS_SECRET_KEY, FYERS_REDIRECT_URI).",
        )
    return True, f"Logged in to Fyers today at {tok.obtained_at:%H:%M} IST."


def login_chip() -> None:
    """Small status line for other pages."""
    ok, text = token_status()
    (st.success if ok else st.warning)(
        text + ("" if ok or ".env" in text else " Open the Login page."), icon="🔑"
    )


def render(c: Ctx) -> None:
    """Render the page."""
    st.title("Fyers login")
    muted("The Fyers token expires every day, so log in once each morning (before 09:10).")
    ok, text = token_status()
    (st.success if ok else st.warning)(text)
    try:
        from quantswing.broker.fyers_auth import build_login_url, login_with_input
        from quantswing.core.config import get_secrets

        s = get_secrets()
        redirect = s.FYERS_REDIRECT_URI
        url = build_login_url(s.FYERS_CLIENT_ID.get_secret_value(), redirect)
    except Exception:
        st.error(
            "Add FYERS_CLIENT_ID, FYERS_SECRET_KEY and FYERS_REDIRECT_URI to the .env file, "
            "then restart the app."
        )
        return

    def done(u: str) -> None:
        login_with_input(u)

    if callback.is_local(redirect):
        st.subheader("One-click login")
        if st.button("🔑 Log in to Fyers", type="primary"):
            callback.start(redirect, done, c.cfg.paper.live.callback_timeout_s)
            webbrowser.open(url)  # the app runs on this PC: opens your default browser
        st.link_button("Open the Fyers login page again ↗", url)

        @st.fragment(run_every=2)
        def status() -> None:
            stt = callback.STATUS["state"]
            if stt == "waiting":
                st.info("Waiting for you to finish logging in on the Fyers page...", icon="⏳")
            elif stt == "ok":
                callback.STATUS["state"] = "shown"
                st.rerun()  # refresh the banner at the top
            elif stt == "shown":
                st.success(callback.STATUS["message"], icon="✅")
            elif stt == "error":
                st.error(callback.STATUS["message"])

        status()
    else:
        st.info(
            "One-click login needs the Fyers app's redirect URL to point at this PC. In the Fyers "
            "developer portal set the redirect URL to http://127.0.0.1:8765/callback, put the same "
            "value as FYERS_REDIRECT_URI in .env, and restart the app. Until then, use paste login.",
            icon="ℹ️",
        )
    with st.expander("Paste login (backup)", expanded=not callback.is_local(redirect)):
        st.link_button("1. Open the Fyers login page ↗", url)
        pasted = st.text_input(
            "2. After logging in, paste the full address of the page you land on", type="password"
        )
        if st.button("3. Save login") and pasted:
            try:
                login_with_input(pasted)
                st.success("Logged in to Fyers for today.")
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
