"""HTTP client for retrieving ChEMBL records."""

from __future__ import annotations

import logging

import re

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import pandas as pd
import requests  # type: ignore[import-untyped]

try:  # pragma: no cover - поддержка импорта без контекста пакета
    from .http_client import CacheConfig, HttpClient
except ImportError:  # pragma: no cover
    from http_client import CacheConfig, HttpClient  # type: ignore[no-redef]


LOGGER = logging.getLogger(__name__)


def _is_not_found_error(exception: requests.HTTPError) -> bool:
    """Return ``True`` when ``exception`` represents a HTTP 404 error.

    Some versions of :mod:`requests` may omit the :class:`requests.Response`
    object when raising :class:`requests.HTTPError`.  In such cases we fall back
    to inspecting the textual representation to determine whether the error
    corresponds to a missing resource.
    """

    response = exception.response
    if response is not None:
        return response.status_code == 404

    message = str(exception)
    if not message or "Not Found" not in message:
        return False
    return bool(re.search(r"\b404\b", message))


def _coerce_bool(value: Any) -> bool | None:
    """Return a boolean representation of ``value`` when possible.

    Args:
        value: The input value which may already be a boolean, a string
            representation, or a numeric flag.

    Returns:
        A boolean value when the input can be interpreted as such, otherwise
        ``None``.
    """

    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(int(value))
    if isinstance(value, str):
        text = value.strip().lower()
        if not text or text in {"null", "none"}:
            return None
        if text in {"true", "t", "1", "yes", "y"}:
            return True
        if text in {"false", "f", "0", "no", "n"}:
            return False
    return None


def _ensure_data_validity_warning(payload: Dict[str, Any]) -> None:
    """Populate ``data_validity_warning`` in ``payload`` when absent.

    The ChEMBL activity endpoint omits the ``data_validity_warning`` flag, even
    when a descriptive warning is provided.  This helper infers a boolean flag
    from the existing fields so downstream consumers can rely on the presence
    of the column.

    Args:
        payload: A JSON dictionary representing an activity record. The
            dictionary is updated in-place.
    """

    warning = _coerce_bool(payload.get("data_validity_warning"))
    if warning is None:
        warning = _coerce_bool(payload.get("data_validity_flag"))

    if warning is not None:
        payload["data_validity_warning"] = warning
        return

    comment = payload.get("data_validity_comment")
    description = payload.get("data_validity_description")

    if isinstance(comment, str) and comment.strip():
        payload["data_validity_warning"] = True
        return
    if isinstance(description, str) and description.strip():
        payload["data_validity_warning"] = True
        return

    payload["data_validity_warning"] = None


@dataclass
class ApiCfg:
    """Minimal configuration for the ChEMBL API."""

    chembl_base: str = "https://www.ebi.ac.uk/chembl/api/data"
    timeout_connect: float = 5.0
    timeout_read: float = 30.0
    user_agent: str = "ChEMBLDataAcquisition/1.0"


@dataclass
class ChemblClient:
    """A client for retrieving data from the ChEMBL API.

    Attributes:
        base_url: The base URL for the ChEMBL API.
        timeout: The timeout for HTTP requests in seconds.
        max_retries: The maximum number of retries for failed requests.
        rps: The number of requests per second to limit to.
        retry_penalty_seconds: Additional delay applied when a ``429`` response
            lacks a ``Retry-After`` hint.
        user_agent: The User-Agent header to use for requests.
        cache_config: Optional configuration for caching HTTP requests.
        http_client: Optional pre-configured HttpClient instance.
    """

    base_url: str = "https://www.ebi.ac.uk/chembl/api/data"
    timeout: float = 30.0
    max_retries: int = 3
    rps: float = 2.0
    retry_penalty_seconds: float = 1.0
    user_agent: str = "ChEMBLDataAcquisition/1.0"
    cache_config: CacheConfig | None = None
    http_client: HttpClient | None = None

    def __post_init__(self) -> None:
        """Initializes the HTTP client."""
        self._http = self.http_client or HttpClient(
            timeout=self.timeout,
            max_retries=self.max_retries,
            rps=self.rps,
            retry_penalty_seconds=self.retry_penalty_seconds,
            cache_config=self.cache_config,
        )

    def _fetch_resource(
        self, resource: str, identifier: str, *, id_field: str
    ) -> Dict[str, Any] | None:
        """Fetches a resource from the ChEMBL API.

        Args:
            resource: The type of resource to fetch (e.g., 'assay', 'activity').
            identifier: The ChEMBL ID of the resource.
            id_field: The name of the ID field in the response JSON.

        Returns:
            A dictionary containing the resource data, or None if not found.
        """
        url = f"{self.base_url.rstrip('/')}/{resource}/{identifier}.json"
        headers = {
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
        try:
            response = self._http.request("get", url, headers=headers)
            response.raise_for_status()
        except requests.HTTPError as exc:
            if _is_not_found_error(exc):
                LOGGER.warning(
                    "%s %s not found (404)", resource.capitalize(), identifier
                )
                return None
            LOGGER.exception("Failed to fetch %s %s", resource, identifier)
            raise
        except requests.RequestException:
            LOGGER.exception("Network error while fetching %s %s", resource, identifier)
            raise

        payload: Dict[str, Any] = response.json()
        payload.setdefault(id_field, identifier)
        return payload

    def fetch_assay(self, assay_id: str) -> Dict[str, Any] | None:
        """Fetches the JSON data for a given assay_id.

        Args:
            assay_id: The ChEMBL ID of the assay.

        Returns:
            A dictionary containing the assay data, or None if not found.
        """
        return self._fetch_resource("assay", assay_id, id_field="assay_chembl_id")

    def fetch_activity(self, activity_id: str) -> Dict[str, Any] | None:
        """Fetches the JSON data for a given activity_id.

        Args:
            activity_id: The ChEMBL ID of the activity.

        Returns:
            A dictionary containing the activity data, or None if not found.
        """
        payload = self._fetch_resource(
            "activity", activity_id, id_field="activity_chembl_id"
        )
        if payload is not None:
            _ensure_data_validity_warning(payload)
        return payload

    def fetch_molecule(self, molecule_id: str) -> Dict[str, Any] | None:
        """Fetches the JSON data for a given molecule_id.

        Args:
            molecule_id: The ChEMBL ID of the molecule.

        Returns:
            A dictionary containing the molecule data, or None if not found.
        """
        return self._fetch_resource(
            "molecule", molecule_id, id_field="molecule_chembl_id"
        )

    def _fetch_many(
        self,
        identifiers: Iterable[str],
        fetcher: Callable[[str], Dict[str, Any] | None],
    ) -> List[Dict[str, Any]]:
        """Fetches multiple resources from the ChEMBL API.

        Args:
            identifiers: An iterable of ChEMBL IDs.
            fetcher: The function to use for fetching each resource.

        Returns:
            A list of dictionaries, where each dictionary contains the data for a resource.
        """
        records: List[Dict[str, Any]] = []
        for identifier in identifiers:
            payload = fetcher(identifier)
            if payload is not None:
                records.append(payload)
        return records

    def fetch_many(self, assay_ids: Iterable[str]) -> List[Dict[str, Any]]:
        """Fetches multiple assay payloads.

        Args:
            assay_ids: An iterable of assay ChEMBL IDs.

        Returns:
            A list of dictionaries, where each dictionary contains the data for an assay.
        """
        return self._fetch_many(assay_ids, self.fetch_assay)

    def fetch_many_activities(
        self, activity_ids: Iterable[str]
    ) -> List[Dict[str, Any]]:
        """Fetches multiple activity payloads.

        Args:
            activity_ids: An iterable of activity ChEMBL IDs.

        Returns:
            A list of dictionaries, where each dictionary contains the data for an activity.
        """
        return self._fetch_many(activity_ids, self.fetch_activity)

    def fetch_many_molecules(self, molecule_ids: Iterable[str]) -> List[Dict[str, Any]]:
        """Fetches multiple molecule payloads.

        Args:
            molecule_ids: An iterable of molecule ChEMBL IDs.

        Returns:
            A list of dictionaries, where each dictionary contains the data for a molecule.
        """
        return self._fetch_many(molecule_ids, self.fetch_molecule)

    def request_json(self, url: str, *, cfg: ApiCfg, timeout: float) -> Dict[str, Any]:
        """Return the JSON payload for ``url`` using the HTTP client.

        Args:
            url: The URL to request.
            cfg: The API configuration.
            timeout: The read timeout for the request in seconds.

        Returns:
            A dictionary containing the JSON payload. When the resource is not
            found (HTTP 404) an empty dictionary is returned after emitting a
            warning.

        Raises:
            ValueError: If the response cannot be decoded as JSON.
            requests.HTTPError: If the response indicates an HTTP error other
                than 404.
            requests.RequestException: If a network error occurs while making
                the request.
        """

        headers = {
            "Accept": "application/json",
            "User-Agent": cfg.user_agent or self.user_agent,
        }
        timeout_connect = getattr(cfg, "timeout_connect", timeout)
        request_timeout = (timeout_connect, timeout)

        try:
            resp = self._http.request(
                "get", url, timeout=request_timeout, headers=headers
            )
        except requests.RequestException:
            LOGGER.exception("Network error while fetching %s", url)
            raise

        if resp.status_code == 404:
            LOGGER.warning("ChEMBL returned 404 for %s", url)
            return {}

        try:
            resp.raise_for_status()
        except requests.HTTPError:
            LOGGER.exception("HTTP error while fetching %s", url)
            raise

        content_type = resp.headers.get("Content-Type")
        if content_type and "json" not in content_type.lower():
            LOGGER.warning(
                "Unexpected Content-Type %r while fetching %s", content_type, url
            )
            raise ValueError(
                "Expected JSON response from ChEMBL but received" f" {content_type!r}"
            )

        try:
            return resp.json()
        except ValueError as exc:
            LOGGER.warning("Failed to decode JSON payload from %s: %s", url, exc)
            raise


def _chunked(values: Iterable[str], size: int) -> Iterator[List[str]]:
    """Yield ``values`` in lists of length up to ``size``.

    Args:
        values: An iterable of pre-filtered identifier strings.
        size: The desired batch size.

    Yields:
        Lists containing up to ``size`` identifiers preserving the source order.
    """

    if size <= 0:
        raise ValueError("size must be a positive integer")

    chunk: List[str] = []
    for value in values:
        chunk.append(value)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


DOCUMENT_COLUMNS = [
    "document_chembl_id",
    "title",
    "abstract",
    "doi",
    "year",
    "journal",
    "journal_abbrev",
    "volume",
    "issue",
    "first_page",
    "last_page",
    "pubmed_id",
    "authors",
    "source",
]


def _iter_unique_identifiers(ids: Iterable[str]) -> Iterator[str]:
    """Yield unique, non-empty identifiers from ``ids`` preserving order."""

    invalid_markers = {"", "#N/A"}
    seen: set[str] = set()
    for raw in ids:
        text = str(raw)
        if text in invalid_markers:
            continue
        if text in seen:
            continue
        seen.add(text)
        yield text


def get_documents(
    ids: Iterable[str],
    *,
    cfg: ApiCfg,
    client: ChemblClient,
    chunk_size: int = 5,
    timeout: float | None = None,
) -> Iterator[pd.DataFrame]:
    """Stream document records for ``ids`` from the ChEMBL API.

    Args:
        ids: An iterable (including generators) yielding document identifiers.
        cfg: The API configuration.
        client: The ChEMBL client.
        chunk_size: The number of IDs to include in each API request.
        timeout: Optional timeout override for the request in seconds.

    Yields:
        DataFrame chunks containing the retrieved document records. Each chunk
        includes the columns listed in :data:`DOCUMENT_COLUMNS`.
    """

    base = f"{cfg.chembl_base.rstrip('/')}/document.json?format=json"
    effective_timeout = timeout if timeout is not None else cfg.timeout_read

    for chunk in _chunked(_iter_unique_identifiers(ids), chunk_size):
        # Align the ``limit`` parameter with the chunk size so the API returns
        # all requested records even when the server-side default is smaller.
        limit = len(chunk)
        ids_param = ",".join(chunk)
        url = f"{base}&document_chembl_id__in={ids_param}&limit={limit}"
        data = client.request_json(url, cfg=cfg, timeout=effective_timeout)
        items = data.get("documents") or data.get("document") or []
        records: list[dict[str, Any]] = []
        for item in items:
            record = {
                "document_chembl_id": item.get("document_chembl_id"),
                "title": item.get("title"),
                "abstract": item.get("abstract"),
                "doi": item.get("doi"),
                "year": item.get("year"),
                "journal": item.get("journal_full_title"),
                "journal_abbrev": item.get("journal"),
                "volume": item.get("volume"),
                "issue": item.get("issue"),
                "first_page": item.get("first_page"),
                "last_page": item.get("last_page"),
                "pubmed_id": item.get("pubmed_id"),
                "authors": item.get("authors"),
                "source": "ChEMBL",
            }
            records.append(record)
        if not records:
            continue
        df = pd.DataFrame(records)
        yield df.reindex(columns=DOCUMENT_COLUMNS)


__all__ = ["ApiCfg", "ChemblClient", "get_documents"]
