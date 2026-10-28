"""Throttler: token buckets, daily cap/reset, persistence, backoff and retry."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import requests
from hypothesis import given, settings
from hypothesis import strategies as st

from quantswing.broker.throttler import RetryableHTTPError, Throttler, TokenBucket
from quantswing.core.clock import IST
from quantswing.core.config import BackoffCfg, ThrottleCfg
from quantswing.core.types import RateLimitExhausted


class FakeTime:
    """Deterministic monotonic clock whose sleep advances time."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def cfg(
    per_second: int = 10, per_minute: int = 200, per_day: int = 100_000, safety: float = 1.0
) -> ThrottleCfg:
    return ThrottleCfg(
        per_second=per_second,
        per_minute=per_minute,
        per_day=per_day,
        safety_factor=safety,
        day_reset_tz="Asia/Kolkata",
        backoff=BackoffCfg(
            max_retries=3,
            base_delay_s=1.0,
            max_delay_s=4.0,
            jitter_s=0.0,
            retryable_http_codes=[429, 503],
            retryable_api_codes=[429, -429],
            fatal_api_codes=[-353],
        ),
    )


def make(
    c: ThrottleCfg, ft: FakeTime, wall: list[datetime] | None = None, **kw: object
) -> Throttler:
    w = wall or [datetime(2026, 9, 25, 10, 0, tzinfo=IST)]
    return Throttler(
        c,
        clock=ft.clock,
        sleep=ft.sleep,
        wall_clock=lambda: w[0],
        rng=random.Random(0),
        **kw,  # type: ignore[arg-type]
    )


def test_effective_limits_apply_safety_factor() -> None:
    th = make(cfg(10, 200, 100_000, safety=0.8), FakeTime())
    assert (th.limit_sec, th.limit_min, th.limit_day) == (8, 160, 80_000)
    assert make(cfg(1, 1, 1, safety=0.1), FakeTime()).limit_sec == 1  # never below 1


def test_burst_then_waits_per_second() -> None:
    ft = FakeTime()
    th = make(cfg(per_second=5), ft)
    for _ in range(5):
        th.acquire()
    assert ft.t == 0.0  # burst of capacity without waiting
    th.acquire()
    assert ft.t == pytest.approx(0.2)  # 1 token at 5/s


def test_per_minute_limit_binds() -> None:
    ft = FakeTime()
    th = make(cfg(per_second=100, per_minute=10), ft)
    for _ in range(11):
        th.acquire()
    assert ft.t == pytest.approx(6.0)  # 11th call waits 60/10 s


@settings(max_examples=40, deadline=None)
@given(per_sec=st.integers(1, 20), per_min=st.integers(1, 300), n=st.integers(1, 400))
def test_never_exceeds_limits_in_any_window(per_sec: int, per_min: int, n: int) -> None:
    ft = FakeTime()
    th = make(cfg(per_second=per_sec, per_minute=per_min), ft)
    stamps: list[float] = []
    for _ in range(n):
        th.acquire()
        stamps.append(ft.t)
    # Sliding windows: bucket allows `capacity` burst + refill; count within any window
    # of length W must not exceed capacity + W*rate (= 2*capacity for W = window).
    for i, t0 in enumerate(stamps):
        in_sec = sum(1 for t in stamps[i:] if t < t0 + 1.0)
        in_min = sum(1 for t in stamps[i:] if t < t0 + 60.0)
        assert in_sec <= 2 * per_sec
        assert in_min <= 2 * per_min


def test_daily_cap_raises_and_resets_at_ist_midnight() -> None:
    ft = FakeTime()
    wall = [datetime(2026, 9, 25, 23, 59, tzinfo=IST)]
    th = make(cfg(per_day=3), ft, wall)
    for _ in range(3):
        th.acquire()
    with pytest.raises(RateLimitExhausted):
        th.acquire()
    wall[0] = wall[0] + timedelta(minutes=2)  # next IST day
    th.acquire()
    assert th.usage["count"] == 1


def test_day_count_persists(tmp_path: Path) -> None:
    f = tmp_path / "usage.json"
    th = make(cfg(), FakeTime(), state_file=f)
    th.acquire()
    th.acquire()
    th2 = make(cfg(), FakeTime(), state_file=f)
    assert th2.day_count == 2


def test_backoff_delay_exponential_and_capped() -> None:
    th = make(cfg(), FakeTime())
    assert [th.backoff_delay(a) for a in range(5)] == [1.0, 2.0, 4.0, 4.0, 4.0]


def test_call_retries_api_rate_limit_then_succeeds() -> None:
    ft = FakeTime()
    th = make(cfg(), ft)
    responses = iter(
        [{"s": "error", "code": 429}, {"s": "error", "code": -429}, {"s": "ok", "x": 1}]
    )
    out = th.call(lambda: next(responses))
    assert out == {"s": "ok", "x": 1}
    assert ft.sleeps[-2:] == [1.0, 2.0] or ft.sleeps == [1.0, 2.0]


def test_call_retries_exceptions_and_http_then_raises() -> None:
    ft = FakeTime()
    th = make(cfg(), ft)
    calls = {"n": 0}

    def boom() -> dict[str, str]:
        calls["n"] += 1
        raise requests.ConnectionError("down")

    with pytest.raises(requests.ConnectionError):
        th.call(boom)
    assert calls["n"] == 4  # 1 + max_retries

    def http503() -> None:
        raise RetryableHTTPError(503)

    with pytest.raises(RetryableHTTPError):
        th.call(http503)


def test_non_retryable_http_raises_immediately() -> None:
    th = make(cfg(), FakeTime())
    n = {"c": 0}

    def http400() -> None:
        n["c"] += 1
        raise RetryableHTTPError(400)

    with pytest.raises(RetryableHTTPError):
        th.call(http400)
    assert n["c"] == 1


def test_fatal_daily_code_raises() -> None:
    th = make(cfg(), FakeTime())
    with pytest.raises(RateLimitExhausted):
        th.call(
            lambda: {"s": "error", "code": -353, "message": "API Limit exceeded overall per day"}
        )


def test_non_retryable_api_error_returned_to_caller() -> None:
    th = make(cfg(), FakeTime())
    resp = {"s": "error", "code": -300, "message": "invalid symbol"}
    assert th.call(lambda: resp) is resp


def test_token_bucket_refill() -> None:
    ft = FakeTime()
    b = TokenBucket(2, 1.0, ft.clock)
    b.take()
    b.take()
    assert b.wait_time() == pytest.approx(0.5)
    ft.t += 0.5
    assert b.wait_time() == 0.0
