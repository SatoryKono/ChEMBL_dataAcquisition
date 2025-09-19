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

from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from importlib import import_module
import logging
from pathlib import Path
import threading
from types import ModuleType, TracebackType
import time

from threading import Lock, RLock
from typing import Any, Iterable, Mapping, Tuple, cast


import requests  # type: ignore[import-untyped]

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tenacity.wait import wait_base

LOGGER = logging.getLogger(__name__)


DEFAULT_STATUS_FORCELIST: frozenset[int] = frozenset(
    {408, 409, 429, 500, 502, 503, 504}
)


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


def retry_after_from_response(response: requests.Response) -> float | None:
    """Extract the retry delay advertised by ``response`` when available."""

    retry_after = _parse_retry_after(response.headers.get("Retry-After"))
    if retry_after is not None:
        return retry_after

    reset_header = response.headers.get("X-RateLimit-Reset")
    if not reset_header:
        return None
    try:
        reset_epoch = float(reset_header)
    except ValueError:
        return None
    delay = reset_epoch - time.time()
    if delay <= 0:
        return None
    return delay


def _retry_after_from_response(response: requests.Response) -> float | None:
    """Backward compatible wrapper for :func:`retry_after_from_response`."""

    return retry_after_from_response(response)


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
            return retry_after_from_response(exception.response)
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
    cached_session_factory = cast(Any, requests_cache_module).CachedSession
    session = cached_session_factory(
        cache_name=str(cache_path),
        backend="sqlite",
        expire_after=int(cache_config.ttl_seconds),
        allowable_methods=("GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"),
    )
    return session


@dataclass
class RateLimiter:
    """Simple token bucket style rate limiter with penalty support.

    Parameters
    ----------
    rps:
        Maximum number of requests per second. ``0`` disables rate limiting.
    last_call:
        Timestamp of the last request, used to enforce the delay.
    blocked_until:
        Absolute monotonic timestamp until which requests are paused. This is
        used to respect server-side rate limit hints such as ``Retry-After``.
    """

    rps: float
    last_call: float = 0.0
    blocked_until: float = 0.0
    lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)

    def wait(self) -> None:
        """Sleep just enough to satisfy the configured rate limit."""

        sleep_for = 0.0
        while True:
            with self.lock:
                now = time.monotonic()
                wait_until = self.blocked_until if self.blocked_until > now else now

                if self.rps > 0:
                    interval = 1.0 / self.rps
                    next_allowed = self.last_call + interval
                    if next_allowed > wait_until:
                        wait_until = next_allowed

                if wait_until > now:
                    sleep_for = wait_until - now
                else:
                    current_time = max(now, time.monotonic())
                    self.last_call = current_time
                    if self.blocked_until < current_time:
                        self.blocked_until = current_time
                    return

            if sleep_for <= 0:
                continue
            time.sleep(sleep_for)

    def apply_penalty(self, delay_seconds: float | None) -> None:
        """Delay the next request by ``delay_seconds`` when positive.

        Parameters
        ----------
        delay_seconds:
            Duration of the cool-down window. Non-positive values are ignored
            so callers may pass parsed headers verbatim without additional
            validation.
        """

        if not delay_seconds or delay_seconds <= 0:
            return
        target = time.monotonic() + delay_seconds
        with self.lock:
            if target > self.blocked_until:
                self.blocked_until = target


class HttpClient:
    """A wrapper around :mod:`requests` with support for retries, caching, and rate limiting.

    Args:
        timeout: Default request timeout. Can be either a single float
            applied to the connect and read phases, or a tuple of
            ``(connect, read)`` timeouts.
        max_retries: Maximum number of retries for transient errors.
        rps: Target requests per second, implemented via a simple token bucket.
        status_forcelist: HTTP status codes that trigger a retry. Defaults to
            ``DEFAULT_STATUS_FORCELIST``, which excludes ``404 Not Found``.
            To extend the default set, pass a custom iterator, e.g.,
            ``DEFAULT_STATUS_FORCELIST | {404}`` to retry ``404`` responses
            in specific scenarios (e.g., when dealing with unstable indexes).
        backoff_multiplier: Multiplier for the exponential backoff delay between retries.
        retry_penalty_seconds: Additional delay for future requests after receiving a
            ``429 Too Many Requests`` response, if the server did not provide a
            ``Retry-After`` header.
        cache_config: Optional configuration for HTTP request caching.
        session: Optional :class:`requests.Session` to use, allowing for a
            pre-configured session (e.g., with caching or custom headers).
    """

    def __init__(
        self,
        *,
        timeout: float | Tuple[float, float],
        max_retries: int,
        rps: float,
        status_forcelist: Iterable[int] | None = None,
        backoff_multiplier: float = 1.0,
        retry_penalty_seconds: float = 0.0,
        cache_config: "CacheConfig | None" = None,
        session: "requests.Session | None" = None,
    ) -> None:
        if isinstance(timeout, tuple):
            self.timeout = timeout
        else:
            self.timeout = (timeout, timeout)
        self.max_retries = max_retries
        self.rate_limiter = RateLimiter(rps)
        self._cache_config = cache_config
        self._thread_local = threading.local()
        self._session_lock = Lock()
        self._owned_sessions: list[requests.Session] = []
        self._shared_session = session
        if status_forcelist is None:
            self.status_forcelist = set(DEFAULT_STATUS_FORCELIST)
        else:
            self.status_forcelist = set(status_forcelist)
        self.backoff_multiplier = backoff_multiplier
        self.retry_penalty_seconds = retry_penalty_seconds

    @property
    def session(self) -> requests.Session:
        """Return the thread-local :class:`requests.Session` instance."""

        if self._shared_session is not None:
            return self._shared_session
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = create_http_session(self._cache_config)
            setattr(self._thread_local, "session", session)
            with self._session_lock:
                self._owned_sessions.append(session)
        return session

    def close(self) -> None:
        """Close any HTTP sessions owned by this client."""

        if self._shared_session is not None:
            return
        with self._session_lock:
            sessions = list(self._owned_sessions)
            self._owned_sessions.clear()
        for session in sessions:
            try:
                session.close()
            except AttributeError:  # pragma: no cover - defensive
                continue
        self._thread_local = threading.local()

    def __enter__(self) -> "HttpClient":
        """Return ``self`` to support ``with`` statements."""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close owned sessions when leaving a context manager block."""

        self.close()

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
                retry_after = retry_after_from_response(resp)
                if retry_after is not None:
                    LOGGER.warning(
                        "Transient HTTP %s for %s %s; retrying after %.2f seconds",
                        resp.status_code,
                        method.upper(),
                        url,
                        retry_after,
                    )
                    self.rate_limiter.apply_penalty(retry_after)
                else:
                    penalty = None
                    if resp.status_code == 429 and self.retry_penalty_seconds > 0:
                        penalty = self.retry_penalty_seconds
                        self.rate_limiter.apply_penalty(penalty)
                    if penalty:
                        LOGGER.warning(
                            "Transient HTTP %s for %s %s; retrying after %.2f seconds",
                            resp.status_code,
                            method.upper(),
                            url,
                            penalty,
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


__all__ = [
    "CacheConfig",
    "DEFAULT_STATUS_FORCELIST",
    "HttpClient",
    "RateLimiter",
    "RetryAfterWaitStrategy",
    "retry_after_from_response",
    "create_http_session",
]
