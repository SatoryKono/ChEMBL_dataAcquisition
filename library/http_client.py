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
from typing import Any

import requests  # type: ignore[import-untyped]
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
    """Wrapper around :mod:`requests` with retry and rate limiting."""

    def __init__(self, *, timeout: float, max_retries: int, rps: float) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limiter = RateLimiter(rps)
        self.session = requests.Session()

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


__all__ = ["HttpClient", "RateLimiter"]
