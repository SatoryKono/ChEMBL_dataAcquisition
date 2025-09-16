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
from typing import Any, Iterable, Tuple

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

LOGGER = logging.getLogger(__name__)


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
    """Wrapper around :mod:`requests` with retry and rate limiting.

    Parameters
    ----------
    timeout:
        Default timeout for requests. Either a single float applied to both the
        connect and read phases or a ``(connect, read)`` tuple.
    max_retries:
        Maximum number of attempts for transient failures.
    rps:
        Target requests per second enforced via a lightweight token bucket.
    status_forcelist:
        HTTP status codes that should trigger a retry. The default covers the
        most common transient responses from the supported public APIs.
    backoff_multiplier:
        Base multiplier used for exponential backoff between retries.
    session:
        Optional :class:`requests.Session` allowing callers to supply a
        pre-configured session (for example with caching or custom headers).
    """

    def __init__(
        self,
        *,
        timeout: float | Tuple[float, float],
        max_retries: int,
        rps: float,
        status_forcelist: Iterable[int] | None = None,
        backoff_multiplier: float = 1.0,
        session: requests.Session | None = None,
    ) -> None:
        if isinstance(timeout, tuple):
            self.timeout = timeout
        else:
            self.timeout = (timeout, timeout)
        self.max_retries = max_retries
        self.rate_limiter = RateLimiter(rps)
        self.session = session or requests.Session()
        self.status_forcelist = set(status_forcelist or {404, 408, 409, 429, 500, 502, 503, 504})
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

        @retry(
            reraise=True,
            retry=retry_if_exception_type(requests.RequestException),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=self.backoff_multiplier),
        )
        def _do_request() -> requests.Response:
            self.rate_limiter.wait()
            timeout = kwargs.pop("timeout", self.timeout)
            LOGGER.debug("HTTP %s %s (timeout=%s)", method.upper(), url, timeout)
            resp = self.session.request(method, url, timeout=timeout, **kwargs)
            if resp.status_code in self.status_forcelist:
                LOGGER.warning(
                    "Transient HTTP %s for %s %s", resp.status_code, method.upper(), url
                )
                resp.raise_for_status()
            return resp

        return _do_request()


__all__ = ["HttpClient", "RateLimiter"]
