"""Client utilities for retrieving ChEMBL tissue metadata.

The :func:`fetch_tissue_record` helper exposes a small, well-typed API for
querying the public ChEMBL REST endpoint that serves tissue information.

Algorithm Notes
---------------
1. Normalise the provided ChEMBL identifier (trim whitespace and upper-case).
2. Compose the REST endpoint URL using the configured base URL.
3. Issue an HTTP GET request with retry and rate limiting via
   :class:`~library.http_client.HttpClient`.
4. Decode the JSON payload and validate the presence of required fields.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Dict, Iterable, List, Sequence

import requests

try:  # pragma: no cover - fallback for test environments
    from .http_client import CacheConfig, HttpClient
except ImportError:  # pragma: no cover
    from http_client import CacheConfig, HttpClient  # type: ignore[no-redef]

LOGGER = logging.getLogger(__name__)

_TISSUE_ID_PATTERN = re.compile(r"^CHEMBL\d+$")


@dataclass
class TissueConfig:
    """Configuration for accessing the ChEMBL tissue endpoint.

    Parameters
    ----------
    base_url:
        Base URL of the ChEMBL API.
    timeout_sec:
        Timeout (in seconds) applied to individual HTTP requests.
    max_retries:
        Number of retry attempts for transient HTTP errors (>=500 status
        codes or network issues).
    rps:
        Maximum requests per second enforced via a token bucket rate limiter.
    cache:
        Optional :class:`CacheConfig` enabling persistent HTTP caching.
    required_fields:
        Iterable of fields expected to be present in the JSON payload. Missing
        fields result in a :class:`ValueError`.
    """

    base_url: str = "https://www.ebi.ac.uk/chembl/api/data"
    timeout_sec: float = 30.0
    max_retries: int = 3
    rps: float = 2.0
    cache: CacheConfig | None = None
    required_fields: Sequence[str] = (
        "tissue_chembl_id",
        "pref_name",
    )

    def build_tissue_url(self, tissue_id: str) -> str:
        """Return the absolute URL for a tissue resource."""

        base = self.base_url.rstrip("/")
        return f"{base}/tissue/{tissue_id}.json"


def create_http_client(config: TissueConfig) -> HttpClient:
    """Returns an HttpClient configured from a TissueConfig object.

    Args:
        config: The TissueConfig object.

    Returns:
        An HttpClient instance.
    """

    return HttpClient(
        timeout=config.timeout_sec,
        max_retries=config.max_retries,
        rps=config.rps,
        cache_config=config.cache,
    )


class TissueNotFoundError(RuntimeError):
    """Raised when the requested tissue record does not exist."""


def normalise_tissue_id(value: str) -> str:
    """Returns a canonical, upper-case tissue identifier.

    Args:
        value: A raw identifier that may include leading/trailing whitespace or be in a
            lower-case representation.

    Returns:
        The normalized tissue identifier in upper-case form.

    Raises:
        ValueError: If the value is empty after stripping whitespace or does not
            match the expected pattern.
    """

    if value is None:
        raise ValueError("Tissue identifier must not be None")
    tissue_id = value.strip().upper()
    if not tissue_id:
        raise ValueError("Tissue identifier must not be empty")
    if not _TISSUE_ID_PATTERN.match(tissue_id):
        raise ValueError("Tissue identifier must match the pattern 'CHEMBL<digits>'")
    return tissue_id


def _validate_payload(payload: Dict[str, object], *, required: Iterable[str]) -> None:
    """Ensure the decoded payload contains ``required`` fields."""

    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(
            "Missing required fields in tissue payload: " + ", ".join(sorted(missing))
        )


def _fetch_tissue(
    chembl_id: str, config: TissueConfig, client: HttpClient
) -> Dict[str, object]:
    """Return the decoded payload for a single ChEMBL tissue identifier."""

    url = config.build_tissue_url(chembl_id)
    LOGGER.debug("Requesting tissue %s from %s", chembl_id, url)
    response = client.request("get", url, headers={"Accept": "application/json"})
    if response.status_code == requests.codes.not_found:
        raise TissueNotFoundError(f"Tissue record {chembl_id!r} not found")
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:  # pragma: no cover - exercised via integration
        LOGGER.error("ChEMBL tissue request for %s failed: %s", chembl_id, exc)
        raise
    try:
        payload = response.json()
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError("Failed to decode tissue JSON payload") from exc
    _validate_payload(payload, required=config.required_fields)
    payload_id = str(payload.get("tissue_chembl_id", "")).upper()
    if payload_id and payload_id != chembl_id:
        raise ValueError(
            "Tissue identifier mismatch: requested %s but received %s"
            % (chembl_id, payload_id)
        )
    return payload


def fetch_tissue_record(
    chembl_id: str,
    *,
    config: TissueConfig | None = None,
    client: HttpClient | None = None,
) -> Dict[str, object]:
    """Retrieves metadata for a single tissue from ChEMBL.

    Args:
        chembl_id: The ChEMBL identifier for the tissue (case-insensitive).
            Whitespace is stripped, and the value is normalized to upper-case.
        config: An optional TissueConfig object to override connection settings.
        client: An optional pre-configured HttpClient. If omitted, a new client
            is created based on the provided or default config.

    Returns:
        A dictionary representing the parsed JSON payload for the tissue.
    """

    tissue_id = normalise_tissue_id(chembl_id)
    cfg = config or TissueConfig()
    http_client = client or create_http_client(cfg)
    return _fetch_tissue(tissue_id, cfg, http_client)


def fetch_tissues(
    ids: Sequence[str],
    *,
    config: TissueConfig | None = None,
    client: HttpClient | None = None,
    continue_on_error: bool = False,
) -> List[Dict[str, object]]:
    """Retrieves multiple tissue records in sequence.

    Args:
        ids: A sequence of ChEMBL tissue identifiers. Duplicates are ignored while
            preserving input order.
        config: An optional TissueConfig instance.
        client: An optional shared HttpClient.
        continue_on_error: If True, any ValueError, TissueNotFoundError, or
            requests.RequestException is logged and skipped instead of aborting
            the entire retrieval process.

    Returns:
        A list of dictionaries, where each dictionary represents a tissue record.
    """

    cfg = config or TissueConfig()
    http_client = client or create_http_client(cfg)
    results: List[Dict[str, object]] = []
    seen: set[str] = set()
    for raw in ids:
        try:
            tissue_id = normalise_tissue_id(raw)
        except ValueError as exc:
            if continue_on_error:
                LOGGER.warning("Skipping invalid tissue identifier %r: %s", raw, exc)
                continue
            raise
        if tissue_id in seen:
            LOGGER.debug("Skipping duplicate tissue identifier %s", tissue_id)
            continue
        seen.add(tissue_id)
        try:
            results.append(_fetch_tissue(tissue_id, cfg, http_client))
        except TissueNotFoundError as exc:
            if continue_on_error:
                LOGGER.warning("%s", exc)
                continue
            raise
        except requests.RequestException as exc:
            LOGGER.error("Network error while fetching %s: %s", tissue_id, exc)
            if continue_on_error:
                continue
            raise
        except ValueError as exc:
            LOGGER.error("Invalid payload for %s: %s", tissue_id, exc)
            if continue_on_error:
                continue
            raise
    return results


__all__ = [
    "TissueConfig",
    "TissueNotFoundError",
    "create_http_client",
    "fetch_tissue_record",
    "fetch_tissues",
    "normalise_tissue_id",
]
