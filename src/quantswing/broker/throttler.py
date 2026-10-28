"""Central Fyers API throttler: token buckets (sec/min/day) + exponential backoff.

Every Fyers REST call in QuantSwing goes through ONE :class:`Throttler` instance
(06 §1, 02 §2). Limits come from ``settings.throttle`` and are scaled by
``safety_factor``:

    effective_limit = max(1, floor(limit × safety_factor))

Buckets
-------
* per-second and per-minute: continuous-refill token buckets
  (capacity = effective limit, refill rate = limit / window).
* per-day: hard counter that resets at IST midnight (Fyers does not document the reset
  clock — assumption recorded in 07 open items). Optionally persisted to a JSON file so
  restarts during the day keep the count (feeds the Data Health screen later).

Backoff
-------
:meth:`Throttler.call` acquires a token, runs the function, and retries transient
failures with delay ``min(max_delay, base × 2^attempt) + U(0, jitter)``. Transient =
exception of a retryable type, or a Fyers dict response whose ``code`` is in
``retryable_api_codes``. A code in ``fatal_api_codes`` (daily cap) raises
:class:`RateLimitExhausted` immediately.
"""

from __future__ import annotations

import json
import math
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, TypeVar
from zoneinfo import ZoneInfo

import requests

from quantswing.core.config import AppConfig, ThrottleCfg, get_config
from quantswing.core.logging import get_logger
from quantswing.core.types import RateLimitExhausted

log = get_logger(__name__)
T = TypeVar("T")

SECOND = 1.0
MINUTE = 60.0
#: Float tolerance for "one whole token available" (avoids sub-ulp waits that never advance).
TOKEN_EPS = 1e-9

RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    requests.ConnectionError,
    requests.Timeout,
    ConnectionError,
    TimeoutError,
)


class RetryableHTTPError(Exception):
    """Raise from a wrapped call to signal a retryable HTTP status."""

    def __init__(self, status: int, body: str = "") -> None:
        super().__init__(f"HTTP {status}: {body[:200]}")
        self.status = status


@dataclass
class TokenBucket:
    """Continuous-refill token bucket.

    ``capacity`` tokens max; refills at ``capacity / window_s`` tokens per second.
    """

    capacity: int
    window_s: float
    clock: Callable[[], float]
    tokens: float = field(init=False)
    updated: float = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = float(self.capacity)
        self.updated = self.clock()

    @property
    def rate(self) -> float:
        """Refill rate (tokens / second)."""
        return self.capacity / self.window_s

    def _refill(self) -> None:
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now

    def wait_time(self) -> float:
        """Seconds until one token is available (0 if available now)."""
        self._refill()
        return 0.0 if self.tokens >= 1.0 - TOKEN_EPS else (1.0 - self.tokens) / self.rate

    def take(self) -> None:
        """Consume one token (caller must ensure :meth:`wait_time` is 0)."""
        self._refill()
        self.tokens -= 1.0


class Throttler:
    """Thread-safe API throttler with sec/min/day limits and retrying ``call``.

    Inputs: ``cfg`` (``settings.throttle``); ``state_file`` for day-count persistence;
    ``clock`` (monotonic seconds), ``sleep``, ``wall_clock`` (tz-aware now) and ``rng``
    are injectable for deterministic tests.
    """

    def __init__(
        self,
        cfg: ThrottleCfg,
        *,
        state_file: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        wall_clock: Callable[[], datetime] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.cfg = cfg
        self.tz = ZoneInfo(cfg.day_reset_tz)
        self.clock = clock
        self.sleep = sleep
        self.wall_clock = wall_clock or (lambda: datetime.now(tz=self.tz))
        self.rng = rng or random.Random()
        self.state_file = state_file
        self._lock = threading.Lock()

        self.limit_sec = self.effective(cfg.per_second)
        self.limit_min = self.effective(cfg.per_minute)
        self.limit_day = self.effective(cfg.per_day)
        self.sec = TokenBucket(self.limit_sec, SECOND, clock)
        self.min = TokenBucket(self.limit_min, MINUTE, clock)
        self.day: date = self._today()
        self.day_count = 0
        self._load_state()

    @classmethod
    def from_config(cls, cfg: AppConfig | None = None) -> Throttler:
        """Build the process-wide throttler from ``settings.throttle`` + usage state file."""
        cfg = cfg or get_config()
        p = cfg.settings.paths
        return cls(cfg.settings.throttle, state_file=p.resolve(p.api_usage_file))

    # ------------------------------------------------------------------ limits
    def effective(self, limit: int) -> int:
        """``max(1, floor(limit × safety_factor))``."""
        return max(1, math.floor(limit * self.cfg.safety_factor))

    def _today(self) -> date:
        return self.wall_clock().astimezone(self.tz).date()

    def _roll_day(self) -> None:
        today = self._today()
        if today != self.day:
            self.day, self.day_count = today, 0

    # ------------------------------------------------------------------ persistence
    def _load_state(self) -> None:
        if self.state_file is None or not self.state_file.exists():
            return
        try:
            st = json.loads(self.state_file.read_text(encoding="utf-8"))
            if date.fromisoformat(st["day"]) == self.day:
                self.day_count = int(st["count"])
        except (ValueError, KeyError, OSError) as e:
            log.warning("throttler state unreadable; starting fresh", extra={"error": str(e)})

    def _save_state(self) -> None:
        if self.state_file is None:
            return
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"day": self.day.isoformat(), "count": self.day_count}))
        tmp.replace(self.state_file)

    # ------------------------------------------------------------------ acquire
    def acquire(self) -> None:
        """Block until a request slot is free in every bucket, then consume it.

        Raises :class:`RateLimitExhausted` if the (effective) daily cap is used up.
        """
        while True:
            with self._lock:
                self._roll_day()
                if self.day_count >= self.limit_day:
                    raise RateLimitExhausted(
                        f"daily API cap reached ({self.day_count}/{self.limit_day})"
                    )
                wait = max(self.sec.wait_time(), self.min.wait_time())
                if wait <= 0:
                    self.sec.take()
                    self.min.take()
                    self.day_count += 1
                    self._save_state()
                    return
            self.sleep(wait)

    @property
    def usage(self) -> dict[str, Any]:
        """Snapshot of today's usage (for logs / Data Health screen)."""
        return {"day": self.day.isoformat(), "count": self.day_count, "day_limit": self.limit_day}

    # ------------------------------------------------------------------ backoff
    def backoff_delay(self, attempt: int) -> float:
        """Delay before retry ``attempt`` (0-based): ``min(max, base·2^a) + U(0, jitter)``."""
        b = self.cfg.backoff
        expo: float = b.base_delay_s * float(2**attempt)
        return float(min(b.max_delay_s, expo) + self.rng.uniform(0.0, b.jitter_s))

    def _api_code(self, resp: Any) -> int | None:
        if isinstance(resp, dict) and resp.get("s") == "error":
            code = resp.get("code")
            try:
                return int(code) if code is not None else None
            except (TypeError, ValueError):
                return None
        return None

    def call(self, fn: Callable[[], T], *, label: str = "api") -> T:
        """Throttled call of ``fn`` with retry on transient failures.

        Returns ``fn()``'s result. Raises the last error after ``max_retries`` retries.
        """
        b = self.cfg.backoff
        attempt = 0
        while True:
            self.acquire()
            try:
                result = fn()
            except RetryableHTTPError as e:
                if e.status not in b.retryable_http_codes or attempt >= b.max_retries:
                    raise
                reason: str = f"http {e.status}"
            except RETRYABLE_EXCEPTIONS as e:
                if attempt >= b.max_retries:
                    raise
                reason = type(e).__name__
            else:
                code = self._api_code(result)
                if code is not None and code in b.fatal_api_codes:
                    raise RateLimitExhausted(f"{label}: fatal API code {code}", result)
                if code is None or code not in b.retryable_api_codes:
                    return result
                if attempt >= b.max_retries:
                    return result  # caller sees the error response
                reason = f"api code {code}"
            delay = self.backoff_delay(attempt)
            log.warning(
                "retrying after transient failure",
                extra={"label": label, "attempt": attempt + 1, "reason": reason, "delay_s": delay},
            )
            self.sleep(delay)
            attempt += 1
