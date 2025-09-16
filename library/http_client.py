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
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from importlib import import_module
import logging
from pathlib import Path
from types import ModuleType
import time

from typing import Any, Callable, Iterable, Mapping, Tuple, cast


import requests

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tenacity.wait import wait_base

LOGGER = logging.getLogger(__name__)


def _parse_retry_after(value: str | None) -> float | None:
    """Return retry delay in seconds parsed from a ``Retry-After`` header."""

    if value is None:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    try:
        seconds = float(candidate)
    except ValueError:
        try:
            retry_dt = parsedate_to_datetime(candidate)
        except (TypeError, ValueError):
            return None
        if retry_dt.tzinfo is None:
            retry_dt = retry_dt.replace(tzinfo=timezone.utc)
        delta = (retry_dt - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, delta)
    else:
        return max(0.0, seconds)


def _retry_after_from_response(response: requests.Response) -> float | None:
    """Extract the retry delay from ``response`` when advertised by the server."""

    return _parse_retry_after(response.headers.get("Retry-After"))


class RetryAfterWaitStrategy(wait_base):
    """Tenacity wait strategy that honours ``Retry-After`` headers."""

    def __init__(self, fallback: wait_base) -> None:
        self._fallback = fallback

    def __call__(self, retry_state: RetryCallState) -> float:
        retry_after = self._retry_after_seconds(retry_state)
        if retry_after is not None:
            if retry_after > 0:
                LOGGER.debug(
                    "Retry-After header requested sleeping for %.2f seconds",
                    retry_after,
                )
                return retry_after
            LOGGER.debug("Retry-After header requested immediate retry")
            return 0.0
        return self._fallback(retry_state)

    @staticmethod
    def _retry_after_seconds(retry_state: RetryCallState) -> float | None:
        if retry_state.outcome is None or not retry_state.outcome.failed:
            return None
        exception = retry_state.outcome.exception()
        if isinstance(exception, requests.HTTPError) and exception.response is not None:
            return _retry_after_from_response(exception.response)
        return None


def _import_requests_cache() -> ModuleType | None:
    """Return the :mod:`requests_cache` module when installed.

    The HTTP cache is optional because some environments (for example,
    lightweight Windows installations) may execute the CLI utilities without the
    ``requests-cache`` dependency pre-installed.  Returning ``None`` allows the
    caller to gracefully fall back to an uncached session while emitting a
    helpful log message that explains how to enable caching.
    """

    try:
        return import_module("requests_cache")
    except ModuleNotFoundError:
        return None


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

    A plain session is returned when caching is disabled, misconfigured or the
    optional ``requests-cache`` dependency is unavailable.  Otherwise,
    :class:`requests_cache.CachedSession` is instantiated with the provided
    settings.
    """

    if cache_config is None or not cache_config.is_active():
        return requests.Session()
    requests_cache_module = _import_requests_cache()
    if requests_cache_module is None:
        LOGGER.warning(
            "HTTP caching requested but optional dependency 'requests-cache' is "
            "missing. Install it via 'pip install requests-cache' to enable "
            "caching. Proceeding without caching."
        )
        return requests.Session()

    assert cache_config.path is not None
    cache_path = Path(cache_config.path).expanduser()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.debug(
        "HTTP cache enabled at %s with TTL %.0f s",
        cache_path,
        cache_config.ttl_seconds,
    )
    cached_session_factory = cast(
        Callable[..., requests.Session],
        getattr(requests_cache_module, "CachedSession"),
    )
    session = cached_session_factory(
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
    """Обёртка вокруг :mod:`requests` с поддержкой повторов, кэширования и ограничения скорости.

    Parameters
    ----------
    timeout:
        Таймаут по умолчанию для запросов. Может быть либо одним числом (float), применяемым к фазам соединения и чтения, либо кортежем ``(connect, read)``.
    max_retries:
        Максимальное количество попыток при временных ошибках.
    rps:
        Целевое количество запросов в секунду, реализуемое через простой token bucket.
    status_forcelist:
        Коды HTTP-статусов, при которых будет выполняться повтор запроса. По умолчанию включает наиболее распространённые временные ошибки.
    backoff_multiplier:
        Множитель для экспоненциальной задержки между попытками.
    cache_config:
        Необязательная конфигурация кэширования HTTP-запросов.
    session:
        Необязательный :class:`requests.Session`, позволяющий использовать заранее сконфигурированную сессию (например, с кэшем или кастомными заголовками).
    """

    def __init__(
        self,
        *,
        timeout: float | Tuple[float, float],
        max_retries: int,
        rps: float,
        status_forcelist: Iterable[int] | None = None,
        backoff_multiplier: float = 1.0,
        cache_config: "CacheConfig | None" = None,
        session: "requests.Session | None" = None,
    ) -> None:
        if isinstance(timeout, tuple):
            self.timeout = timeout
        else:
            self.timeout = (timeout, timeout)
        self.max_retries = max_retries
        self.rate_limiter = RateLimiter(rps)
        if session is not None:
            self.session = session
        else:
            # Если передан cache_config, используем create_http_session, иначе обычную сессию
            self.session = create_http_session(cache_config)
        self.status_forcelist = set(
            status_forcelist or {404, 408, 409, 429, 500, 502, 503, 504}
        )
        self.backoff_multiplier = backoff_multiplier

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

        Returns
        -------
        :class:`requests.Response`
            Raw HTTP response. Callers are responsible for decoding JSON/XML
            payloads and interpreting non-2xx statuses that do not trigger a
            retry.
        """

        wait_strategy = RetryAfterWaitStrategy(
            wait_exponential(multiplier=self.backoff_multiplier)
        )

        @retry(
            reraise=True,
            retry=retry_if_exception_type(requests.RequestException),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_strategy,
        )
        def _do_request() -> requests.Response:
            self.rate_limiter.wait()
            timeout = kwargs.pop("timeout", self.timeout)
            LOGGER.debug("HTTP %s %s (timeout=%s)", method.upper(), url, timeout)
            resp = self.session.request(method, url, timeout=timeout, **kwargs)
            if resp.status_code in self.status_forcelist:
                retry_after = _retry_after_from_response(resp)
                if retry_after is not None:
                    LOGGER.warning(
                        "Transient HTTP %s for %s %s; retrying after %.2f seconds",
                        resp.status_code,
                        method.upper(),
                        url,
                        retry_after,
                    )
                else:
                    LOGGER.warning(
                        "Transient HTTP %s for %s %s",
                        resp.status_code,
                        method.upper(),
                        url,
                    )
                resp.raise_for_status()
            return resp

        return _do_request()


__all__ = ["CacheConfig", "HttpClient", "RateLimiter", "create_http_session"]
