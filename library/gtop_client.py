"""HTTP client for the IUPHAR/BPS Guide to PHARMACOLOGY (GtoPdb) API.

The :class:`GtoPClient` wraps :class:`~library.http_client.HttpClient` to provide
convenient helpers for resolving target identifiers and fetching related
resources.  The client only performs network requests and leaves normalisation to
:mod:`gtop_normalize`.

Algorithm Notes
---------------
1. All methods return Python data structures decoded from the JSON responses.
2. HTTP 204 responses are interpreted as ``None``/empty collections rather than
   raising errors.
3. The :func:`resolve_target` function performs deterministic selection of the
   target record, preferring Human entries when multiple species are returned.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional

from http_client import HttpClient

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
    """

    base_url: str
    timeout_sec: float = 30.0
    max_retries: int = 3
    rps: float = 2.0


class GtoPClient:
    """Client for the GtoPdb REST API."""

    def __init__(self, cfg: GtoPConfig) -> None:
        self.cfg = cfg
        self.http = HttpClient(
            timeout=cfg.timeout_sec, max_retries=cfg.max_retries, rps=cfg.rps
        )
        self.base_url = cfg.base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Low level request helper

    def _get(self, path: str, params: Dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        LOGGER.debug("GET %s params=%s", url, params)
        resp = self.http.request("get", url, params=params)
        if resp.status_code in (204, 404):
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
        """Search ``/targets`` using one of the supported parameters."""

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
        """Fetch a child endpoint for a specific ``target_id``."""

        payload = self._get(f"/targets/{target_id}/{endpoint}", params=params) or []
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
