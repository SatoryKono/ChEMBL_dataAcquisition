from __future__ import annotations

from pathlib import Path
import threading
import time

import pytest
import requests  # type: ignore[import-untyped]

from library.http_client import (
    CacheConfig,
    DEFAULT_STATUS_FORCELIST,
    HttpClient,
    RateLimiter,
    create_http_session,
)


class FakeClock:
    """Deterministic clock used to capture sleep durations in tests."""

    def __init__(self) -> None:
        self.monotonic_time = 0.0
        self.wall_time = 1_000_000.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.monotonic_time += seconds
        self.wall_time += seconds

    def monotonic(self) -> float:
        return self.monotonic_time

    def time(self) -> float:
        return self.wall_time


def test_http_client_uses_cache(tmp_path: Path, requests_mock) -> None:
    """Responses are served from cache on repeated calls."""

    url = "https://example.org/resource"
    requests_mock.get(url, json={"status": "ok"})
    cache_path = tmp_path / "http_cache"
    client = HttpClient(
        timeout=1,
        max_retries=1,
        rps=0,
        cache_config=CacheConfig(enabled=True, path=str(cache_path), ttl_seconds=60),
    )

    first = client.request("get", url)
    assert first.json() == {"status": "ok"}
    assert not getattr(first, "from_cache", False)

    second = client.request("get", url)
    assert second.json() == {"status": "ok"}
    assert getattr(second, "from_cache", False)

    assert requests_mock.call_count == 1


def test_create_http_session_falls_back_when_cache_dependency_missing(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    """A plain :class:`requests.Session` is returned when caching dependency is absent."""

    caplog.set_level("WARNING", "library.http_client")
    monkeypatch.setattr("library.http_client._import_requests_cache", lambda: None)
    cache_path = tmp_path / "http_cache"
    config = CacheConfig(enabled=True, path=str(cache_path), ttl_seconds=60)

    session = create_http_session(config)

    assert isinstance(session, requests.Session)
    assert "requests-cache" in caplog.text


def _patch_clock(monkeypatch, clock: FakeClock) -> None:
    """Patch time related functions to use ``clock`` for deterministic sleeps."""

    monkeypatch.setattr("tenacity.nap.time.sleep", clock.sleep)
    monkeypatch.setattr("library.http_client.time.sleep", clock.sleep)
    monkeypatch.setattr("library.http_client.time.monotonic", clock.monotonic)
    monkeypatch.setattr("library.http_client.time.time", clock.time)


def test_http_client_honours_retry_after(monkeypatch, requests_mock) -> None:
    """The client sleeps according to ``Retry-After`` headers before retrying."""

    clock = FakeClock()
    _patch_clock(monkeypatch, clock)
    url = "https://example.org/throttled"
    requests_mock.post(
        url,
        [
            {
                "status_code": 429,
                "headers": {"Retry-After": "3"},
                "json": {"detail": "rate limit"},
            },
            {"status_code": 200, "json": {"status": "ok"}},
        ],
    )
    client = HttpClient(timeout=1, max_retries=2, rps=0)

    response = client.request("post", url)

    assert response.json() == {"status": "ok"}
    assert clock.sleeps
    assert pytest.approx(3.0, rel=1e-3) == clock.sleeps[0]


def test_http_client_falls_back_when_retry_after_invalid(
    monkeypatch, requests_mock
) -> None:
    """Invalid ``Retry-After`` values defer to exponential backoff."""

    clock = FakeClock()
    _patch_clock(monkeypatch, clock)
    url = "https://example.org/throttled-invalid"
    requests_mock.get(
        url,
        [
            {
                "status_code": 429,
                "headers": {"Retry-After": "not-a-date"},
                "json": {"detail": "rate limit"},
            },
            {"status_code": 200, "json": {"status": "ok"}},
        ],
    )
    client = HttpClient(timeout=1, max_retries=2, rps=0, backoff_multiplier=0.5)

    response = client.request("get", url)

    assert response.json() == {"status": "ok"}
    assert clock.sleeps
    assert pytest.approx(0.5, rel=1e-3) == clock.sleeps[0]


def test_http_client_respects_rate_limit_reset_header(
    monkeypatch, requests_mock
) -> None:
    """``X-RateLimit-Reset`` headers act as ``Retry-After`` equivalents."""

    clock = FakeClock()
    _patch_clock(monkeypatch, clock)
    url = "https://example.org/rate-limit-reset"
    requests_mock.get(
        url,
        [
            {
                "status_code": 429,
                "headers": {"X-RateLimit-Reset": str(clock.time() + 30)},
                "json": {"detail": "rate limit"},
            },
            {"status_code": 200, "json": {"status": "ok"}},
        ],
    )
    client = HttpClient(timeout=1, max_retries=2, rps=0)

    response = client.request("get", url)

    assert response.json() == {"status": "ok"}
    assert pytest.approx(30.0, rel=1e-3) == clock.sleeps[0]


def test_http_client_does_not_retry_not_found_by_default(requests_mock) -> None:
    """404 responses are returned to the caller instead of being retried."""

    url = "https://example.org/missing"
    requests_mock.get(
        url,
        [
            {"status_code": 404, "json": {"detail": "missing"}},
            {"status_code": 200, "json": {"status": "ok"}},
        ],
    )
    client = HttpClient(timeout=1, max_retries=2, rps=0)

    response = client.request("get", url)

    assert response.status_code == 404
    assert response.json() == {"detail": "missing"}
    assert requests_mock.call_count == 1


def test_http_client_retries_opt_in_not_found_status(
    monkeypatch, requests_mock
) -> None:
    """404 responses can be retried when explicitly configured."""

    clock = FakeClock()
    _patch_clock(monkeypatch, clock)
    url = "https://example.org/flaky-index"
    requests_mock.get(
        url,
        [
            {"status_code": 404, "json": {"detail": "pending"}},
            {"status_code": 200, "json": {"status": "ok"}},
        ],
    )
    client = HttpClient(
        timeout=1,
        max_retries=2,
        rps=0,
        status_forcelist=DEFAULT_STATUS_FORCELIST | {404},
    )

    response = client.request("get", url)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert requests_mock.call_count == 2
    assert clock.sleeps


def test_http_client_penalises_future_requests(monkeypatch, requests_mock) -> None:
    """Fallback penalty slows down subsequent retries without ``Retry-After``."""

    clock = FakeClock()
    _patch_clock(monkeypatch, clock)
    url = "https://example.org/hard-limit"
    requests_mock.post(
        url,
        [
            {"status_code": 429, "json": {"detail": "limit"}},
            {"status_code": 200, "json": {"status": "ok"}},
        ],
    )
    client = HttpClient(
        timeout=1,
        max_retries=2,
        rps=0,
        retry_penalty_seconds=5.0,
    )

    response = client.request("post", url)

    assert response.json() == {"status": "ok"}
    # Tenacity waits ``backoff_multiplier`` seconds (defaults to 1.0)
    # followed by the penalty applied by the rate limiter.
    assert pytest.approx([1.0, 4.0], rel=1e-3) == clock.sleeps


def test_rate_limiter_penalty_blocks_until_elapsed(monkeypatch) -> None:
    """Rate limiter delays requests until the configured penalty expires."""

    clock = FakeClock()
    monkeypatch.setattr("library.http_client.time.sleep", clock.sleep)
    monkeypatch.setattr("library.http_client.time.monotonic", clock.monotonic)
    limiter = RateLimiter(rps=0)

    limiter.apply_penalty(2.5)
    limiter.wait()

    assert pytest.approx(2.5, rel=1e-3) == clock.monotonic_time
    # Subsequent waits without additional penalties do not sleep again.
    limiter.wait()
    assert pytest.approx(2.5, rel=1e-3) == clock.monotonic_time


def test_rate_limiter_serialises_updates_across_threads() -> None:
    """Concurrent ``wait`` calls honour the configured requests-per-second cap."""

    limiter = RateLimiter(rps=50)
    thread_count = 5
    per_thread_calls = 10
    total_calls = thread_count * per_thread_calls
    timestamps: list[float] = []
    timestamp_lock = threading.Lock()
    start_barrier = threading.Barrier(thread_count + 1)

    def worker() -> None:
        start_barrier.wait()
        for _ in range(per_thread_calls):
            limiter.wait()
            stamp = time.perf_counter()
            with timestamp_lock:
                timestamps.append(stamp)

    threads = [threading.Thread(target=worker) for _ in range(thread_count)]
    for thread in threads:
        thread.start()

    start_barrier.wait()
    for thread in threads:
        thread.join()

    assert len(timestamps) == total_calls
    ordered = sorted(timestamps)
    elapsed = ordered[-1] - ordered[0]
    assert elapsed > 0

    expected_min = (total_calls - 1) / limiter.rps
    # Allow a small 10% tolerance to account for scheduling variance.
    assert elapsed >= expected_min * 0.9
