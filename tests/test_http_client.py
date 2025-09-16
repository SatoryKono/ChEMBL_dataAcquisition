from __future__ import annotations

from pathlib import Path

from library.http_client import CacheConfig, HttpClient


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
