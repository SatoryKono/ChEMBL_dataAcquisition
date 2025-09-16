"""HTTP client for the IUPHAR/BPS Guide to PHARMACOLOGY (GtoPdb) API.

The :class:`GtoPClient` wraps :class:`~library.http_client.HttpClient` to provide
convenient helpers for resolving target identifiers and fetching related
resources.  The client only performs network requests and leaves normalisation to
:mod:`gtop_normalize`.

Algorithm Notes
---------------
1. All methods return Python data structures decoded from the JSON responses.
2. HTTP 204, 400 and 404 responses are interpreted as ``None``/empty
   collections. HTTP 400 responses are logged at debug level because the
   service occasionally uses this status code to signal an empty result set.
3. The :func:`resolve_target` function performs deterministic selection of the
   target record, preferring Human entries when multiple species are returned.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional

import requests  # type: ignore[import-untyped]

# ``gtop_client`` is imported both as a module within the package and directly in
# tests. The conditional import below supports both patterns.
try:  # pragma: no cover - support test environments
    from .http_client import CacheConfig, HttpClient
except ImportError:  # pragma: no cover
    from http_client import CacheConfig, HttpClient  # type: ignore[no-redef]


LOGGER = logging.getLogger(__name__)


@dataclass
class GtoPConfig:
    """Network and API configuration for :class:`GtoPClient`.

    Parameters
    ----------
    base_url:
        Base URL of the web service, e.g. ``"https://www.guidetopharmacology.org/services"``.
    timeout_sec:
        Network timeout applied to each request.
    max_retries:
        Number of retry attempts for transient failures.
    rps:
        Maximum requests per second enforced via a token bucket.
    cache:
        Optional HTTP cache configuration shared by all requests.
    """

    base_url: str
    timeout_sec: float = 30.0
    max_retries: int = 3
    rps: float = 2.0
    cache: CacheConfig | None = None


class GtoPClient:
    """Client for the GtoPdb REST API."""

    def __init__(self, cfg: GtoPConfig) -> None:
        self.cfg = cfg
        self.http = HttpClient(
            timeout=cfg.timeout_sec,
            max_retries=cfg.max_retries,
            rps=cfg.rps,
            cache_config=cfg.cache,
        )
        self.base_url = cfg.base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Low level request helper

    def _get(self, path: str, params: Dict[str, Any] | None = None) -> Any:
        """Perform a GET request to a GtoPdb API endpoint.

        This is a low-level helper that handles common status codes and JSON
        parsing.

        Parameters
        ----------
        path:
            The API path to request (e.g., "/targets").
        params:
            A dictionary of query parameters.

        Returns
        -------
        Any
            The parsed JSON response, or None if the response is empty or an
            error occurred.
        """
        url = f"{self.base_url}{path}"
        LOGGER.debug("GET %s params=%s", url, params)
        resp = self.http.request("get", url, params=params)
        if resp.status_code in (204, 404):
            return None
        if resp.status_code == 400:
            # The API occasionally returns HTTP 400 for queries that simply
            # yield no results. Treat this as an empty payload so that the
            # pipeline can continue processing other targets without
            # emitting noisy warnings.
            LOGGER.debug("HTTP 400 for %s params=%s", url, params)
            return None
        resp.raise_for_status()
        if not resp.content:
            return None
        return resp.json()

    # ------------------------------------------------------------------
    # Public API

    def search_targets(
        self,
        *,
        accession: str | None = None,
        database: str | None = None,
        gene_symbol: str | None = None,
        name: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Search for targets using various criteria.

        This method queries the `/targets` endpoint of the GtoPdb API.

        Parameters
        ----------
        accession:
            An accession number from a supported database.
        database:
            The database for the accession number (e.g., "UniProt", "HGNC").
        gene_symbol:
            A gene symbol.
        name:
            A target name.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries, each representing a matching target.
        """

        params: Dict[str, Any] = {}
        if accession:
            params["accession"] = accession
            if database:
                params["database"] = database
        if gene_symbol:
            params["geneSymbol"] = gene_symbol
        if name:
            params["name"] = name
        payload = self._get("/targets", params=params) or []
        return payload

    def fetch_target_endpoint(
        self, target_id: int, endpoint: str, params: Dict[str, Any] | None = None
    ) -> List[Dict[str, Any]]:
        """Fetch data from a specific endpoint related to a target.

        For example, this can be used to get interactions or other detailed
        information for a given target ID.

        Parameters
        ----------
        target_id:
            The GtoPdb ID of the target.
        endpoint:
            The name of the endpoint to fetch (e.g., "interactions").
        params:
            Optional query parameters.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries from the endpoint's response.
            If the GtoP service responds with a 4xx (400) or 5xx
            status code the issue is logged and an empty list is
            returned so that downstream processing can continue.
        """
        try:
            payload = self._get(f"/targets/{target_id}/{endpoint}", params=params) or []
        except requests.HTTPError as exc:  # pragma: no cover - network fallback
            status_code: int | None = None
            if exc.response is not None:
                status_code = exc.response.status_code
            if status_code == 400:
                LOGGER.warning(
                    "GtoP request failed for %s/%s with params %s: %s",
                    target_id,
                    endpoint,
                    params,
                    exc.response.text,
                )
                return []
            if status_code is not None and 500 <= status_code < 600:
                LOGGER.error(
                    "GtoP server error for %s/%s with params %s: %s",
                    target_id,
                    endpoint,
                    params,
                    exc.response.text if exc.response is not None else str(exc),
                )
                return []
            raise
        return payload


# ---------------------------------------------------------------------------
# Identifier resolution


def resolve_target(
    client: GtoPClient, identifier: str, id_column: str
) -> Optional[Dict[str, Any]]:
    """Resolve ``identifier`` to a single target record.

    Parameters
    ----------
    client:
        Instance of :class:`GtoPClient` used for network requests.
    identifier:
        Raw identifier value extracted from the input file.
    id_column:
        Column name describing the identifier type. One of
        ``"uniprot_id"``, ``"target_name"``, ``"hgnc_id"`` or
        ``"gene_symbol"``.

    Returns
    -------
    dict or None
        Resolved target object or ``None`` if no match could be found.
    """

    identifier = identifier.strip()
    targets: List[Dict[str, Any]]
    if id_column == "uniprot_id":
        targets = client.search_targets(
            accession=identifier.upper(), database="UniProt"
        )
    elif id_column == "hgnc_id":
        acc = identifier.upper()
        if not acc.startswith("HGNC:"):
            acc = f"HGNC:{acc}"
        targets = client.search_targets(accession=acc, database="HGNC")
    elif id_column == "gene_symbol":
        targets = client.search_targets(gene_symbol=identifier)
    elif id_column == "target_name":
        targets = client.search_targets(name=identifier)
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported id column: {id_column}")

    if not targets:
        LOGGER.warning("No target found for %s=%s", id_column, identifier)
        return None

    human = [t for t in targets if t.get("species") == "Human"]
    if human:
        return human[0]
    targets.sort(key=lambda t: int(t.get("targetId", 0)))
    return targets[0]


__all__ = ["GtoPClient", "GtoPConfig", "resolve_target"]
