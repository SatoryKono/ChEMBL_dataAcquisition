from __future__ import annotations

from pathlib import Path

import pytest
import requests  # type: ignore[import-untyped]

from library.http_client import CacheConfig, HttpClient, create_http_session


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


def test_http_client_honours_retry_after(monkeypatch, requests_mock) -> None:
    """The client sleeps according to ``Retry-After`` headers before retrying."""

    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("tenacity.nap.time.sleep", fake_sleep)
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
    assert sleeps
    assert pytest.approx(3.0, rel=1e-3) == sleeps[0]


def test_http_client_falls_back_when_retry_after_invalid(
    monkeypatch, requests_mock
) -> None:
    """Invalid ``Retry-After`` values defer to exponential backoff."""

    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("tenacity.nap.time.sleep", fake_sleep)
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
    assert sleeps
    assert pytest.approx(0.5, rel=1e-3) == sleeps[0]
