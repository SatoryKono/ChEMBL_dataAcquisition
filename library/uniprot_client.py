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

from dataclasses import dataclass, field
import json
import logging
import time
from typing import Any, Dict, Iterable, List, Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

LOGGER = logging.getLogger(__name__)


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
    """Thin wrapper around the UniProtKB REST API."""

    base_url: str
    fields: str
    network: NetworkConfig
    rate_limit: RateLimitConfig
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:  # pragma: no cover - simple initialisation
        self.base_url = self.base_url.rstrip("/")
        self._last_call = 0.0

    # ------------------------------------------------------------------
    def _wait_rate_limit(self) -> None:
        """Sleep if necessary to enforce the configured rate limit."""
        if self.rate_limit.rps <= 0:
            return
        interval = 1.0 / self.rate_limit.rps
        now = time.monotonic()
        delta = now - self._last_call
        if delta < interval:
            time.sleep(interval - delta)
        self._last_call = time.monotonic()

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
        @retry(
            reraise=True,
            retry=retry_if_exception_type(requests.RequestException),
            stop=stop_after_attempt(self.network.max_retries),
            wait=wait_exponential(multiplier=self.network.backoff_sec),
        )
        def _do_request() -> requests.Response:
            self._wait_rate_limit()
            LOGGER.debug("GET %s params=%s", url, params)
            resp = self.session.get(
                url, params=params, timeout=self.network.timeout_sec
            )
            if resp.status_code >= 500:
                resp.raise_for_status()
            return resp

        try:
            resp = _do_request()
        except requests.RequestException:  # pragma: no cover - network failure
            LOGGER.error("Request for %s failed after all retries", url, exc_info=True)
            raise
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
