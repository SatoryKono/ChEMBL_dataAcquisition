"""Lightweight HTTP client with retry and rate limiting.

This module exposes the :class:`HttpClient` which wraps :mod:`requests` to
provide a deterministic network layer for the command line utilities in this
repository.  The client honours a maximum request rate and performs
exponential backoff retries on transient errors.

Algorithm Notes
---------------
1. Before each outgoing request the client checks the configured requests per
   second limit and sleeps if necessary.
2. HTTP errors and network issues are retried up to ``max_retries`` times with
   exponential backoff.
3. Responses are returned as :class:`requests.Response` objects and are
   expected to be decoded by the caller.

"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from pathlib import Path
from typing import Any, Mapping, Optional


import requests
import requests_cache

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for persistent HTTP caching.

    Parameters
    ----------
    enabled:
        Whether the cache layer should be activated.  When ``False`` a regular
        :class:`requests.Session` instance is created.
    path:
        Filesystem path passed to :class:`requests_cache.CachedSession`.  The
        parent directory is created automatically when caching is enabled.
    ttl_seconds:
        Time-to-live for cached responses in seconds.  ``0`` disables
        persistence.
    """

    enabled: bool = False
    path: str | None = None
    ttl_seconds: float = 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "CacheConfig" | None:
        """Build a :class:`CacheConfig` from a configuration mapping.

        Parameters
        ----------
        data:
            Mapping holding configuration keys such as ``enabled``, ``path`` and
            ``ttl``/``ttl_sec``/``ttl_seconds``.

        Returns
        -------
        CacheConfig | None
            Parsed configuration or ``None`` when the mapping is empty.
        """

        if not data:
            return None
        ttl = (
            data.get("ttl")
            or data.get("ttl_sec")
            or data.get("ttl_seconds")
            or data.get("expire_after")
        )
        ttl_value = float(ttl) if ttl is not None else 0.0
        path = data.get("path")
        return cls(
            enabled=bool(data.get("enabled", False)),
            path=str(path) if path is not None else None,
            ttl_seconds=ttl_value,
        )

    def is_active(self) -> bool:
        """Return ``True`` when caching should be enabled."""

        return (
            self.enabled
            and self.path is not None
            and isinstance(self.ttl_seconds, (int, float))
            and self.ttl_seconds > 0
        )


def create_http_session(cache_config: CacheConfig | None = None) -> requests.Session:
    """Return a :class:`requests.Session` honouring ``cache_config``.

    When caching is disabled or misconfigured, a plain session is returned.
    Otherwise, :class:`requests_cache.CachedSession` is instantiated with the
    provided settings.
    """

    if cache_config is None or not cache_config.is_active():
        return requests.Session()
    assert cache_config.path is not None
    cache_path = Path(cache_config.path).expanduser()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.debug(
        "HTTP cache enabled at %s with TTL %.0f s",
        cache_path,
        cache_config.ttl_seconds,
    )
    session = requests_cache.CachedSession(
        cache_name=str(cache_path),
        backend="sqlite",
        expire_after=int(cache_config.ttl_seconds),
        allowable_methods=("GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"),
    )
    return session


@dataclass
class RateLimiter:
    """Simple token bucket style rate limiter.

    Parameters
    ----------
    rps:
        Maximum number of requests per second. ``0`` disables rate limiting.
    last_call:
        Timestamp of the last request, used to enforce the delay.
    """

    rps: float
    last_call: float = 0.0

    def wait(self) -> None:
        """Sleep just enough to satisfy the configured rate limit."""

        if self.rps <= 0:
            return
        interval = 1.0 / self.rps
        now = time.monotonic()
        delta = now - self.last_call
        if delta < interval:
            time.sleep(interval - delta)
        self.last_call = time.monotonic()


class HttpClient:
    """Wrapper around :mod:`requests` with retry, caching and rate limiting."""

    def __init__(
        self,
        *,
        timeout: float,
        max_retries: int,
        rps: float,
        cache_config: CacheConfig | None = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limiter = RateLimiter(rps)
        self.session = session or create_http_session(cache_config)

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        """Perform an HTTP request honouring retry and rate limits.

        Parameters
        ----------
        method:
            HTTP verb such as ``"get"`` or ``"post"``.
        url:
            Absolute URL to request.
        **kwargs:
            Additional arguments passed to :func:`requests.request`.
        """

        @retry(
            reraise=True,
            retry=retry_if_exception_type(requests.RequestException),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1),
        )
        def _do_request() -> requests.Response:
            self.rate_limiter.wait()
            resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
            if resp.status_code >= 500:
                resp.raise_for_status()
            return resp

        resp = _do_request()
        return resp


__all__ = ["CacheConfig", "HttpClient", "RateLimiter", "create_http_session"]
