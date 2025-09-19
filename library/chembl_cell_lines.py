"""Utilities for retrieving ChEMBL cell line records.

Algorithm Notes
---------------
1. Normalise the provided ChEMBL cell line identifier to upper case and
   validate it is not empty.
2. Build the ChEMBL API endpoint URL and perform an HTTP GET request using the
   shared :class:`~library.http_client.HttpClient` with retry and rate limiting.
3. Treat HTTP ``404`` responses as missing records and raise
   :class:`CellLineNotFoundError`.
4. Parse the JSON payload into a Python dictionary and return it to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict

import requests

from .http_client import CacheConfig, HttpClient

LOGGER = logging.getLogger(__name__)

CellLineRecord = Dict[str, Any]


class CellLineError(RuntimeError):
    """Base class for cell line related exceptions."""


class CellLineNotFoundError(CellLineError):
    """Raised when the requested cell line identifier does not exist."""


class CellLineServiceError(CellLineError):
    """Raised when the ChEMBL service returns an unexpected response."""


@dataclass
class CellLineConfig:
    """Configuration controlling network access to the ChEMBL API.

    Parameters
    ----------
    base_url:
        Base URL for the ChEMBL API.
    timeout_sec:
        Timeout in seconds applied to HTTP requests.
    max_retries:
        Maximum number of retry attempts for transient failures.
    rps:
        Maximum number of requests per second. ``0`` disables rate limiting.
    cache:
        Optional cache configuration passed to :class:`HttpClient`.
    """

    base_url: str = "https://www.ebi.ac.uk/chembl/api/data"
    timeout_sec: float = 30.0
    max_retries: int = 3
    rps: float = 2.0
    cache: CacheConfig | None = None


class CellLineClient:
    """A high-level client for fetching ChEMBL cell line records.

    Args:
        config: An optional CellLineConfig object. If not provided, a default
            configuration will be used.
    """

    def __init__(self, config: CellLineConfig | None = None) -> None:
        """Initializes the CellLineClient."""
        self.config = config or CellLineConfig()
        self._http = HttpClient(
            timeout=self.config.timeout_sec,
            max_retries=self.config.max_retries,
            rps=self.config.rps,
            cache_config=self.config.cache,
        )

    def fetch_cell_line(self, cell_chembl_id: str) -> CellLineRecord:
        """Return the raw cell line record for ``cell_chembl_id``.

        Parameters
        ----------
        cell_chembl_id:
            Identifier of the cell line, e.g. ``"CHEMBL3307553"``.

        Returns
        -------
        Dict[str, Any]
            Parsed JSON payload returned by the ChEMBL service.

        Raises
        ------
        ValueError
            If ``cell_chembl_id`` is empty after normalisation.
        CellLineNotFoundError
            When the API responds with HTTP ``404``.
        CellLineServiceError
            When the response cannot be parsed as JSON.
        """

        normalised_id = self._normalise_id(cell_chembl_id)
        url = self._build_url(normalised_id)
        LOGGER.debug("Fetching cell line data", extra={"cell_chembl_id": normalised_id})
        response = self._http.request(
            "get",
            url,
            headers={"Accept": "application/json"},
        )
        if response.status_code == 404:
            LOGGER.info("Cell line %s not found", normalised_id)
            raise CellLineNotFoundError(
                f"Cell line {normalised_id} not found at {url}",
            )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:  # pragma: no cover - network edge case
            raise CellLineServiceError(
                f"ChEMBL returned HTTP status {response.status_code} for {url}",
            ) from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise CellLineServiceError(
                "Unable to decode response body as JSON",
            ) from exc
        if not isinstance(payload, dict):
            raise CellLineServiceError(
                "Unexpected payload structure received from ChEMBL",
            )
        LOGGER.debug(
            "Retrieved cell line data", extra={"cell_chembl_id": normalised_id}
        )
        return payload

    def _build_url(self, cell_chembl_id: str) -> str:
        """Return the API endpoint URL for ``cell_chembl_id``."""

        base = self.config.base_url.rstrip("/")
        return f"{base}/cell_line/{cell_chembl_id}.json"

    @staticmethod
    def _normalise_id(cell_chembl_id: str) -> str:
        """Normalise a raw identifier to upper case and trim whitespace."""

        if cell_chembl_id is None:
            raise ValueError("cell_chembl_id must not be None")
        normalised = str(cell_chembl_id).strip().upper()
        if not normalised:
            raise ValueError("cell_chembl_id must not be empty")
        return normalised


def fetch_cell_line(
    cell_chembl_id: str, config: CellLineConfig | None = None
) -> CellLineRecord:
    """Convenience wrapper returning the record for ``cell_chembl_id``.

    Parameters
    ----------
    cell_chembl_id:
        Identifier of the cell line to fetch.
    config:
        Optional :class:`CellLineConfig` overriding network settings.

    Returns
    -------
    Dict[str, Any]
        JSON dictionary returned by the ChEMBL API.
    """

    client = CellLineClient(config)
    return client.fetch_cell_line(cell_chembl_id)


__all__ = [
    "CellLineClient",
    "CellLineConfig",
    "CellLineError",
    "CellLineNotFoundError",
    "CellLineServiceError",
    "fetch_cell_line",
]
