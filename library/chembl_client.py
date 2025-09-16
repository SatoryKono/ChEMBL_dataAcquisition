"""HTTP client for retrieving ChEMBL records."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Dict, Iterable, List

import requests  # type: ignore[import-untyped]

try:  # pragma: no cover - support importing without package context
    from .http_client import CacheConfig, HttpClient
except ImportError:  # pragma: no cover
    from http_client import CacheConfig, HttpClient  # type: ignore[no-redef]

LOGGER = logging.getLogger(__name__)


@dataclass
class ChemblClient:
    """Client responsible for fetching payloads from the ChEMBL API.

    Parameters
    ----------
    base_url:
        Base URL for the ChEMBL web services.
    timeout:
        Request timeout in seconds.
    max_retries:
        Number of retry attempts for transient network errors.
    rps:
        Maximum number of requests per second; ``0`` disables rate limiting.
    user_agent:
        Value for the ``User-Agent`` header used in outgoing HTTP requests.
    cache_config:
        Optional :class:`CacheConfig` controlling HTTP caching behaviour.
    http_client:
        Pre-configured :class:`HttpClient` instance.  When omitted a new client
        is created using the provided configuration values.
    """

    base_url: str = "https://www.ebi.ac.uk/chembl/api/data"
    timeout: float = 30.0
    max_retries: int = 3
    rps: float = 2.0
    user_agent: str = "ChEMBLDataAcquisition/1.0"
    cache_config: CacheConfig | None = None
    http_client: HttpClient | None = None

    def __post_init__(self) -> None:
        self._http = self.http_client or HttpClient(
            timeout=self.timeout,
            max_retries=self.max_retries,
            rps=self.rps,
            cache_config=self.cache_config,
        )

    def _fetch_resource(
        self, resource: str, identifier: str, *, id_field: str
    ) -> Dict[str, Any] | None:
        url = f"{self.base_url.rstrip('/')}/{resource}/{identifier}.json"
        headers = {
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
        try:
            response = self._http.request("get", url, headers=headers)
            if response.status_code == 404:
                LOGGER.warning(
                    "%s %s was not found (404)", resource.capitalize(), identifier
                )
                return None
            response.raise_for_status()
        except requests.HTTPError:
            LOGGER.exception("Failed to fetch %s %s", resource, identifier)
            raise
        except requests.RequestException:
            LOGGER.exception(
                "Network error while fetching %s %s", resource, identifier
            )
            raise

        payload: Dict[str, Any] = response.json()
        payload.setdefault(id_field, identifier)
        return payload

    def fetch_assay(self, assay_id: str) -> Dict[str, Any] | None:
        """Return the JSON payload for ``assay_id``."""

        return self._fetch_resource("assay", assay_id, id_field="assay_chembl_id")

    def fetch_activity(self, activity_id: str) -> Dict[str, Any] | None:
        """Return the JSON payload for ``activity_id``."""

        return self._fetch_resource(
            "activity", activity_id, id_field="activity_chembl_id"
        )

    def _fetch_many(
        self, identifiers: Iterable[str], fetcher: Callable[[str], Dict[str, Any] | None]
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for identifier in identifiers:
            payload = fetcher(identifier)
            if payload is not None:
                records.append(payload)
        return records

    def fetch_many(self, assay_ids: Iterable[str]) -> List[Dict[str, Any]]:
        """Fetch multiple assays and return the successful payloads."""

        return self._fetch_many(assay_ids, self.fetch_assay)

    def fetch_many_activities(
        self, activity_ids: Iterable[str]
    ) -> List[Dict[str, Any]]:
        """Fetch multiple activities and return the successful payloads."""

        return self._fetch_many(activity_ids, self.fetch_activity)
