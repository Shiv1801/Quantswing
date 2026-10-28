"""D40 one-click Fyers login: a tiny local web server catches the redirect after you log in.

Works when the Fyers app's redirect URL points at this PC, e.g.
``http://127.0.0.1:8765/callback`` (set the same value in the Fyers developer portal and as
``FYERS_REDIRECT_URI`` in ``.env``). The browser is sent there after login with
``?auth_code=...``; the server exchanges it for today's token (09 §1, the same code as
``fyers_login.py``) and shows "Logged in". Listens on 127.0.0.1 only, for one login, then
stops (or after ``timeout`` seconds).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

LOCAL_HOSTS = ("127.0.0.1", "localhost")

# shared by Streamlit reruns (same Python process): state, message, server
STATUS: dict[str, Any] = {"state": "idle", "message": "", "server": None, "since": 0.0}
_LOCK = threading.Lock()

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>QuantSwing login</title>
<style>body{{font-family:system-ui,sans-serif;background:#0f1115;color:#e6e6e6;display:flex;
align-items:center;justify-content:center;height:100vh;margin:0}}div{{text-align:center}}
h1{{font-size:22px}}p{{color:#9aa3b2}}</style></head><body><div><h1>{title}</h1><p>{msg}</p>
</div></body></html>"""


def is_local(redirect_uri: str) -> bool:
    """True if the redirect URL points at this PC (one-click login possible)."""
    u = urlparse(redirect_uri)
    return u.scheme == "http" and (u.hostname or "") in LOCAL_HOSTS and u.port is not None


def start(redirect_uri: str, on_code: Callable[[str], Any], timeout: int = 300) -> bool:
    """Start listening for one login redirect. False if the redirect URL is not local."""
    if not is_local(redirect_uri):
        return False
    u = urlparse(redirect_uri)
    path = u.path or "/"
    with _LOCK:
        old = STATUS.get("server")
        if old is not None and STATUS["state"] == "waiting":
            return True  # already waiting
        if old is not None:
            _shutdown(old)

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                req = urlparse(self.path)
                if req.path.rstrip("/") != path.rstrip("/"):
                    self.send_response(404)
                    self.end_headers()
                    return
                full = f"{u.scheme}://{u.netloc}{self.path}"
                if "auth_code=" not in full:
                    self._page(400, "No login code", "Fyers did not send a login code - try again.")
                    return
                try:
                    on_code(full)
                    STATUS.update(state="ok", message="Logged in to Fyers for today.")
                    self._page(
                        200, "Logged in ✓", "You can close this tab and go back to QuantSwing."
                    )
                except Exception as e:
                    STATUS.update(state="error", message=f"{type(e).__name__}: {e}")
                    self._page(500, "Login failed", str(e)[:300])
                threading.Thread(target=_shutdown, args=(self.server,), daemon=True).start()

            def _page(self, code: int, title: str, msg: str) -> None:
                body = PAGE.format(title=title, msg=msg).encode()
                self.send_response(code)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a: Any) -> None:  # never log URLs (they carry the code)
                return

        try:
            srv = ThreadingHTTPServer(("127.0.0.1", int(u.port or 80)), Handler)
        except OSError as e:
            STATUS.update(state="error", message=f"Port {u.port} is busy: {e}")
            return True
        STATUS.update(
            state="waiting", message="Waiting for the Fyers login...", server=srv, since=time.time()
        )
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        threading.Thread(target=_expire, args=(srv, timeout), daemon=True).start()
    return True


def _expire(srv: ThreadingHTTPServer, timeout: int) -> None:
    time.sleep(timeout)
    if STATUS.get("server") is srv and STATUS["state"] == "waiting":
        STATUS.update(state="error", message="Timed out waiting for the login - click Login again.")
        _shutdown(srv)


def _shutdown(srv: Any) -> None:
    try:
        srv.shutdown()
        srv.server_close()
    except Exception:
        pass
    if STATUS.get("server") is srv:
        STATUS["server"] = None
