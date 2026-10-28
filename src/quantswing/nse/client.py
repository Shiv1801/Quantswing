"""Polite NSE HTTP client (02 §2 scraping etiquette, D31).

* Browser headers + session cookies from a warm-up visit to the NSE site. NSE's API sits
  behind a bot filter that rejects Python's default TLS fingerprint, so ``curl_cffi``
  (browser impersonation) is used when installed; plain ``requests`` otherwise.
* At most one request per ``min_interval_s``; retries with backoff; cookies are refreshed
  on 401/403 and every ``rewarm_every`` API calls.
* Archive downloads return a status: ``ok`` (content), ``miss`` (404 / HTML page / wrong
  file type — e.g. a holiday or a file not published yet) or ``fail`` (blocked / network
  error after retries). A block is never recorded as a miss.
* Windows: the corporate certificate store is exported to a PEM bundle so HTTPS works
  behind SSL inspection (same issue ``pip-system-certs`` solves for ``requests``).
"""

from __future__ import annotations

import json
import ssl
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from quantswing.core.config import NseHttpCfg
from quantswing.core.logging import get_logger

log = get_logger(__name__)

RETRY_CODES = frozenset({401, 403, 408, 429, 500, 502, 503, 504})


class Response(Protocol):
    """The subset of ``requests``/``curl_cffi`` responses used here."""

    status_code: int
    content: bytes
    headers: Any

    def json(self) -> Any: ...


class Session(Protocol):
    """The subset of ``requests``/``curl_cffi`` sessions used here."""

    headers: Any

    def get(self, url: str, **kwargs: Any) -> Response: ...


class NseError(RuntimeError):
    """NSE could not be reached / kept refusing after retries."""


@dataclass(frozen=True)
class Download:
    """Result of an archive download."""

    status: str  # ok | miss | fail
    content: bytes = b""
    url: str = ""
    http: int = 0
    error: str = ""


def headers(cfg: NseHttpCfg) -> dict[str, str]:
    """Browser-like request headers."""
    return {
        "User-Agent": cfg.user_agent,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": cfg.site_url + "/",
    }


def system_ca_bundle(dest: Path) -> Path | None:
    """Windows only: write certifi + OS trust-store CAs to ``dest`` (PEM); else ``None``."""
    if sys.platform != "win32":
        return None
    try:
        import certifi

        pems: list[str] = []
        enum = getattr(ssl, "enum_certificates", None)  # Windows only
        for store in ("ROOT", "CA"):
            for der, enc, trust in enum(store) if enum else []:
                if enc == "x509_asn" and (trust is True or "1.3.6.1.5.5.7.3.1" in trust):
                    pems.append(ssl.DER_cert_to_PEM_cert(der))
        dest.parent.mkdir(parents=True, exist_ok=True)
        base = Path(certifi.where()).read_text(encoding="utf-8")
        dest.write_text(base + "\n" + "".join(pems), encoding="utf-8")
        log.info("system CA bundle written", extra={"system_certs": len(pems)})
        return dest
    except Exception as e:
        log.warning("system CA export failed", extra={"error": repr(e)})
        return None


def system_proxies() -> dict[str, str]:
    """OS proxy settings (Windows registry / env vars) — curl_cffi does not read them itself."""
    import urllib.request

    return {k: v for k, v in urllib.request.getproxies().items() if k in ("http", "https")}


def _verify(cfg: NseHttpCfg, ca_dir: Path | None) -> Any:
    if cfg.use_system_certs and ca_dir is not None:
        bundle = system_ca_bundle(ca_dir / "ca_bundle.pem")
        if bundle is not None:
            return str(bundle)
    return True


def requests_session(cfg: NseHttpCfg) -> Session:
    """Plain ``requests`` session (uses the OS proxy + cert store via pip-system-certs)."""
    import requests

    rs = requests.Session()
    rs.headers.update(headers(cfg))
    return cast(Session, rs)


def make_session(cfg: NseHttpCfg, ca_dir: Path | None = None) -> tuple[Session, str]:
    """(session, backend name). curl_cffi with impersonation if available, else requests."""
    if cfg.impersonate and cfg.impersonate != "none":
        try:
            from curl_cffi import requests as cr

            s: Any = cr.Session(
                impersonate=cfg.impersonate,
                verify=_verify(cfg, ca_dir),
                proxies=cast(Any, system_proxies() or None),
            )
            s.headers.update(headers(cfg))
            return s, "curl_cffi"
        except ImportError:
            log.warning("curl_cffi not installed; NSE API calls may be blocked")
    return requests_session(cfg), "requests"


def looks_like_html(content: bytes) -> bool:
    """True for an HTML page (NSE serves one instead of a missing file)."""
    head = content[:512].lstrip().lower()
    return head.startswith(b"<!doctype") or head.startswith(b"<html") or b"<html" in head


class NseClient:
    """Throttled, cookie-aware NSE client. ``session``/``sleep``/``clock`` are injectable."""

    def __init__(
        self,
        cfg: NseHttpCfg,
        session: Session | None = None,
        *,
        ca_dir: Path | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        if session is None:
            session, self.backend = make_session(cfg, ca_dir)
        else:
            self.backend = "injected"
        self.session = session
        self._sleep, self._clock = sleep, clock
        self._last = -1e18
        self._warm = False
        self._api_calls = 0
        self.requests = 0
        self._answered = 0  # responses received (any HTTP status)
        self._can_fallback = self.backend == "curl_cffi"
        self.last_error = ""

    # ------------------------------------------------------------------ plumbing
    def _wait(self) -> None:
        dt = self.cfg.min_interval_s - (self._clock() - self._last)
        if dt > 0:
            self._sleep(dt)
        self._last = self._clock()

    def _get(self, url: str, params: dict[str, Any] | None = None) -> Response:
        self._wait()
        self.requests += 1
        try:
            r = self.session.get(url, params=params, timeout=self.cfg.timeout_s)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"[:300]
            tls_or_net = any(
                k in self.last_error.lower()
                for k in ("certificate", "ssl", "tls", "connect", "proxy", "resolve")
            )
            if not (self._can_fallback and (self._answered == 0 or tls_or_net)):
                raise
            # curl_cffi cannot connect at all (corporate proxy / SSL inspection):
            # switch to requests, which uses the Windows proxy + certificate store.
            log.warning(
                "curl_cffi failed; falling back to requests", extra={"error": self.last_error}
            )
            self._can_fallback = False
            self.session, self.backend = requests_session(self.cfg), "requests (fallback)"
            self._warm = False
            r = self.session.get(url, params=params, timeout=self.cfg.timeout_s)
        self._answered += 1
        return r

    def warm(self) -> None:
        """Visit the site to obtain session cookies (errors are tolerated)."""
        try:
            self._get(self.cfg.site_url + self.cfg.warmup_path)
            self._warm = True
        except Exception as e:
            log.warning("nse warm-up failed", extra={"error": repr(e)})

    # ------------------------------------------------------------------ archives
    def download(
        self, urls: Sequence[str], magic: bytes | None = None, min_bytes: int = 200
    ) -> Download:
        """First URL that yields a valid file. ``magic`` = required leading bytes (zip: PK)."""
        last_fail: Download | None = None
        for url in urls:
            res = self._download_one(url, magic, min_bytes)
            if res.status == "ok":
                return res
            if res.status == "fail":
                last_fail = res
        return last_fail or Download("miss", url=urls[-1] if urls else "")

    def _download_one(self, url: str, magic: bytes | None, min_bytes: int) -> Download:
        for attempt in range(self.cfg.retries + 1):
            try:
                r = self._get(url)
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"[:300]
                log.warning("nse download error", extra={"url": url, "error": repr(e)})
                code = 0
            else:
                code = int(r.status_code)
                body = bytes(r.content or b"")
                if code == 200:
                    if looks_like_html(body) or len(body) < min_bytes:
                        return Download("miss", url=url, http=code)
                    if magic and not body.startswith(magic):
                        return Download("miss", url=url, http=code)
                    return Download("ok", body, url, code)
                if code in (404, 410):
                    return Download("miss", url=url, http=code)
                if code not in RETRY_CODES:
                    return Download(
                        "fail", url=url, http=code, error=self.last_error if code == 0 else ""
                    )
            if attempt < self.cfg.retries:
                self._sleep(self.cfg.backoff_s * (attempt + 1))
                if code in (401, 403):
                    self.warm()
        return Download("fail", url=url, http=code, error=self.last_error if code == 0 else "")

    # ------------------------------------------------------------------ JSON API
    def api(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """GET ``{api_url}/{endpoint}`` → parsed JSON. Raises :class:`NseError`."""
        if not self._warm or self._api_calls % self.cfg.rewarm_every == 0:
            self.warm()
        self._api_calls += 1
        url = f"{self.cfg.api_url}/{endpoint.lstrip('/')}"
        err = ""
        for attempt in range(self.cfg.retries + 1):
            try:
                r = self._get(url, params)
                code = int(r.status_code)
                if code == 200:
                    try:
                        return r.json()
                    except (ValueError, json.JSONDecodeError):
                        err = "non-JSON response (blocked?)"
                        code = 403
                else:
                    err = f"HTTP {code}"
            except Exception as e:
                err, code = repr(e), 0
            if attempt < self.cfg.retries:
                self._sleep(self.cfg.backoff_s * (attempt + 1))
                if code in (0, 401, 403):
                    self.warm()
        raise NseError(f"{endpoint} {params or ''}: {err}")
