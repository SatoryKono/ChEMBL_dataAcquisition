"""HTTP client for retrieving UniProtKB entries.

The client implements rate limiting and retry logic as configured via
``config.yaml``.  Only a subset of the UniProt REST API is exposed, namely
fetching individual records by accession using the ``/search`` endpoint with an
explicit list of return fields.

Algorithm Notes
---------------
1. Build the request URL for a given accession with the required fields.
2. Honour the configured rate limit before each HTTP request.
3. Retry transient HTTP errors with exponential backoff.
4. Return the parsed JSON document or ``None`` when no result is found.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Dict, Iterable, List, Optional, cast

import requests
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

LOGGER = logging.getLogger(__name__)


try:  # pragma: no cover - optional import paths for tests
    from .http_client import (
        CacheConfig,
        DEFAULT_STATUS_FORCELIST,
        RateLimiter,
        RetryAfterWaitStrategy,
        create_http_session,
        retry_after_from_response,
    )
except ImportError:  # pragma: no cover
    from http_client import (  # type: ignore[no-redef]
        CacheConfig,
        DEFAULT_STATUS_FORCELIST,
        RateLimiter,
        RetryAfterWaitStrategy,
        create_http_session,
        retry_after_from_response,
    )


@dataclass
class NetworkConfig:
    """Network related settings.

    Parameters
    ----------
    timeout_sec:
        Request timeout in seconds.
    max_retries:
        Maximum number of retry attempts for transient failures.
    backoff_sec:
        Exponential backoff multiplier in seconds.
    """

    timeout_sec: float
    max_retries: int
    backoff_sec: float = 1.0


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    rps: float


@dataclass
class UniProtClient:
    """Thin wrapper around the UniProtKB REST API.

    Attributes
    ----------
    base_url:
        Base endpoint for UniProt API requests.
    fields:
        Comma-separated list of fields requested from the search endpoint.
    network:
        Network configuration controlling timeouts and retries.
    rate_limit:
        Rate limiting configuration applied before each request.
    cache:
        Optional HTTP cache configuration applied to outbound requests.
    session:
        Optional :class:`requests.Session` instance. When ``None`` the client
        creates a session honouring ``cache`` automatically.
    """

    base_url: str
    fields: str
    network: NetworkConfig
    rate_limit: RateLimitConfig
    cache: CacheConfig | None = None
    session: Optional[requests.Session] = None

    def __post_init__(self) -> None:  # pragma: no cover - simple initialisation
        self.base_url = self.base_url.rstrip("/")
        self._rate_limiter: RateLimiter = RateLimiter(self.rate_limit.rps)
        self._status_forcelist: set[int] = set(DEFAULT_STATUS_FORCELIST)
        if self.session is None:
            self.session = create_http_session(self.cache)

    def _get_session(self) -> requests.Session:
        """Return the underlying :class:`requests.Session` instance."""

        if self.session is None:  # pragma: no cover - defensive
            raise RuntimeError("HTTP session is not initialised")
        return cast(requests.Session, self.session)

    # ------------------------------------------------------------------
    def _wait_rate_limit(self) -> None:
        """Sleep if necessary to enforce the configured rate limit."""
        self._rate_limiter.wait()

    def _request(self, url: str, params: Dict[str, str]) -> Optional[requests.Response]:
        """Perform a GET request to the UniProt API with retry and rate limiting.

        Parameters
        ----------
        url:
            The URL to request.
        params:
            A dictionary of query parameters.

        Returns
        -------
        Optional[requests.Response]
            The response object, or None if an error occurred.
        """

        session = self._get_session()
        wait_strategy = RetryAfterWaitStrategy(
            wait_exponential(multiplier=self.network.backoff_sec)
        )

        def _log_retry(retry_state: RetryCallState) -> None:
            if retry_state.outcome is None:
                return
            sleep_seconds = 0.0
            if (
                retry_state.next_action is not None
                and retry_state.next_action.sleep is not None
            ):
                sleep_seconds = float(retry_state.next_action.sleep)
            exception = retry_state.outcome.exception()
            reason = "unknown"
            if (
                isinstance(exception, requests.HTTPError)
                and exception.response is not None
            ):
                status_code = exception.response.status_code
                reason = f"HTTP {status_code}"
                if status_code in {408, 429} and sleep_seconds > 0:
                    self._rate_limiter.apply_penalty(sleep_seconds)
            elif exception is not None:
                reason = repr(exception)
            next_attempt = retry_state.attempt_number + 1
            LOGGER.warning(
                "Retrying GET %s (attempt %d/%d) after %.2f seconds due to %s",
                url,
                next_attempt,
                self.network.max_retries,
                sleep_seconds,
                reason,
            )

        @retry(
            reraise=True,
            retry=retry_if_exception_type(requests.RequestException),
            stop=stop_after_attempt(self.network.max_retries),
            wait=wait_strategy,
            before_sleep=_log_retry,
        )
        def _do_request() -> requests.Response:
            self._wait_rate_limit()
            LOGGER.debug("GET %s params=%s", url, params)
            resp = session.get(url, params=params, timeout=self.network.timeout_sec)
            if resp.status_code in self._status_forcelist:
                retry_after = retry_after_from_response(resp)
                if retry_after is not None and retry_after > 0:
                    LOGGER.warning(
                        "Transient HTTP %s for GET %s; server requested %.2f seconds pause",
                        resp.status_code,
                        url,
                        retry_after,
                    )
                    self._rate_limiter.apply_penalty(retry_after)
                else:
                    LOGGER.warning(
                        "Transient HTTP %s for GET %s; retrying with backoff",
                        resp.status_code,
                        url,
                    )
                resp.raise_for_status()
            elif resp.status_code >= 500:
                resp.raise_for_status()
            return resp

        try:
            resp = _do_request()
        except requests.RequestException as exc:  # pragma: no cover - network failure
            LOGGER.warning("Request failed for %s: %s", url, exc)
            return None
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:  # pragma: no cover - unexpected codes
            LOGGER.warning("Unexpected response %s for %s", resp.status_code, url)
            return None
        return resp

    def fetch(self, accession: str) -> Optional[Dict[str, Any]]:
        """Fetch a UniProt entry for ``accession``.

        Parameters
        ----------
        accession:
            UniProt accession to retrieve.

        Returns
        -------
        Optional[Dict[str, Any]]
            Parsed JSON document or ``None`` when not found.
        """

        params = {
            "query": f"accession:{accession}",
            "format": "json",
            "fields": self.fields,
            "size": "1",
        }
        url = f"{self.base_url}/search"
        resp = self._request(url, params)
        if not resp:
            return None
        try:
            data = resp.json()
        except json.JSONDecodeError:  # pragma: no cover - API guarantees JSON
            LOGGER.warning("Invalid JSON for %s", accession)
            return None
        results = data.get("results", [])
        if not results:
            return None
        return results[0]

    # ------------------------------------------------------------------
    def fetch_entry_json(self, accession: str) -> Optional[Dict[str, Any]]:
        """Retrieve the full UniProt entry for ``accession``.

        Parameters
        ----------
        accession:
            UniProt accession to retrieve.

        Returns
        -------
        Optional[Dict[str, Any]]
            Parsed JSON document or ``None`` when not found.
        """

        url = f"{self.base_url}/{accession}.json"
        resp = self._request(url, {})
        if not resp:
            return None
        try:
            return resp.json()
        except json.JSONDecodeError:  # pragma: no cover - API guarantees JSON
            LOGGER.warning("Invalid JSON for %s", accession)
            return None

    def fetch_entries_json(
        self, accessions: Iterable[str], *, batch_size: int = 100
    ) -> Dict[str, Dict[str, Any]]:
        """Retrieve multiple UniProt entries in batches.

        Parameters
        ----------
        accessions:
            Iterable of UniProt accessions to retrieve.
        batch_size:
            Maximum number of accessions to query per HTTP request.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Mapping from accession to the corresponding JSON entry. Missing
            accessions are omitted from the result.
        """

        unique = [a for a in dict.fromkeys(accessions) if a]
        entries: Dict[str, Dict[str, Any]] = {}
        for i in range(0, len(unique), batch_size):
            chunk = unique[i : i + batch_size]
            query = " OR ".join(f"accession:{acc}" for acc in chunk)
            params = {"format": "json", "query": query, "size": str(len(chunk))}
            url = f"{self.base_url}/stream"
            resp = self._request(url, params)
            if not resp:
                continue
            try:
                data = resp.json()
            except json.JSONDecodeError:  # pragma: no cover - API guarantees JSON
                LOGGER.warning("Invalid JSON for %s", ",".join(chunk))
                continue
            for entry in data.get("results", []) or []:
                acc = entry.get("primaryAccession")
                if acc:
                    entries[acc] = entry
        return entries

    def fetch_isoforms_fasta(self, accession: str) -> List[str]:
        """Return FASTA headers for ``accession`` including isoforms.

        The UniProt REST API streams FASTA records when the ``includeIsoform``
        flag is set.  Only the header lines are returned by this method.

        Parameters
        ----------
        accession:
            UniProt accession to retrieve.

        Returns
        -------
        List[str]
            List of FASTA header lines. Empty when the request fails.
        """

        params = {
            "query": f"accession:{accession}",
            "format": "fasta",
            "includeIsoform": "true",
        }
        url = f"{self.base_url}/stream"
        resp = self._request(url, params)
        if not resp:
            return []
        headers: List[str] = []
        for line in resp.text.splitlines():
            if line.startswith(">"):
                headers.append(line)
        return headers
