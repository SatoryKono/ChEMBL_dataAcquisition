from __future__ import annotations

from pathlib import Path

import requests

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
