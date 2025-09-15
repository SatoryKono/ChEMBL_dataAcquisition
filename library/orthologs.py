"""Clients for retrieving gene orthology information.

The module provides two HTTP clients to query orthology relationships from
Ensembl and OMA.  Only a small subset of each API is wrapped and the focus is
on the data required by :mod:`get_uniprot_target_data`.

Algorithm Notes
---------------
1. Map species aliases (``human`` -> ``homo_sapiens``) to the identifiers
   required by the Ensembl REST service.
2. Request orthology information for an Ensembl gene identifier using the
   ``/homology`` endpoint and normalise the result into :class:`Ortholog`
   objects.
3. Optionally resolve gene symbols and UniProt accessions via lookup endpoints.
4. Provide a very small OMA fallback client that currently only returns an
   empty result but keeps the interface extensible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, Iterable, List, Optional

import requests  # type: ignore[import-untyped]
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for static type checking only
    from .uniprot_client import NetworkConfig, RateLimitConfig
else:  # pragma: no cover - allow package or top-level imports
    try:
        from .uniprot_client import NetworkConfig, RateLimitConfig
    except ImportError:  # pragma: no cover
        from uniprot_client import NetworkConfig, RateLimitConfig

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures


@dataclass
class Ortholog:
    """Normalised representation of an orthologous gene."""

    target_species: str
    target_gene_symbol: str
    target_ensembl_gene_id: str
    target_uniprot_id: str | None = None
    homology_type: str | None = None
    perc_id: float | None = None
    perc_pos: float | None = None
    is_high_confidence: bool | None = None
    dn: float | None = None
    ds: float | None = None
    source_db: str = "Ensembl"

    def to_ordered_dict(self) -> Dict[str, Any]:
        """Return a dictionary with alphabetically sorted keys."""

        data: Dict[str, Any] = {
            "dn": self.dn,
            "ds": self.ds,
            "homology_type": self.homology_type or "",
            "is_high_confidence": self.is_high_confidence,
            "perc_id": self.perc_id,
            "perc_pos": self.perc_pos,
            "source_db": self.source_db,
            "target_ensembl_gene_id": self.target_ensembl_gene_id,
            "target_gene_symbol": self.target_gene_symbol,
            "target_species": self.target_species,
            "target_uniprot_id": self.target_uniprot_id or "",
        }
        return {k: data[k] for k in sorted(data.keys())}


# ---------------------------------------------------------------------------
# Helpers

_SPECIES_MAP = {
    "human": "homo_sapiens",
    "mouse": "mus_musculus",
    "rat": "rattus_norvegicus",
    "dog": "canis_familiaris",
    "macaque": "macaca_mulatta",
    "zebrafish": "danio_rerio",
}

_GENE_PREFIX_SPECIES = {
    "ENSG": "homo_sapiens",
    "ENSMUSG": "mus_musculus",
    "ENSRNOG": "rattus_norvegicus",
    "ENSCAFG": "canis_familiaris",
    "ENSMFAG": "macaca_mulatta",
    "ENSDARG": "danio_rerio",
}


def _map_species(name: str) -> str:
    """Map a species alias to its Ensembl name."""
    return _SPECIES_MAP.get(name, name)


def _species_from_gene(ensembl_gene_id: str) -> Optional[str]:
    """Infer the species from an Ensembl gene ID prefix."""
    for prefix, species in _GENE_PREFIX_SPECIES.items():
        if ensembl_gene_id.startswith(prefix):
            return species
    return None


# ---------------------------------------------------------------------------
# Ensembl client


@dataclass
class EnsemblHomologyClient:
    """Client for the Ensembl REST ``/homology`` endpoint."""

    base_url: str
    network: NetworkConfig
    rate_limit: RateLimitConfig
    session: requests.Session = field(default_factory=requests.Session)

    _last_call: float = 0.0
    _symbol_cache: Dict[str, str] = field(default_factory=dict)
    _uniprot_cache: Dict[str, str] = field(default_factory=dict)

    def _wait_rate_limit(self) -> None:
        """Sleep if necessary to enforce the configured rate limit."""
        if self.rate_limit.rps <= 0:
            return
        import time

        interval = 1.0 / self.rate_limit.rps
        now = time.monotonic()
        delta = now - self._last_call
        if delta < interval:
            time.sleep(interval - delta)
        self._last_call = time.monotonic()

    def _request(
        self, url: str, params: Iterable[tuple[str, str]]
    ) -> Optional[requests.Response]:
        """Perform a GET request to the Ensembl API.

        This method handles rate limiting and basic error handling.

        Parameters
        ----------
        url:
            The URL to request.
        params:
            A list of (key, value) pairs for the query string.

        Returns
        -------
        Optional[requests.Response]
            The response object, or None if an error occurred.
        """
        self._wait_rate_limit()
        try:
            resp = self.session.get(
                url,
                params=params,
                timeout=self.network.timeout_sec,
                headers={"Accept": "application/json"},
            )
        except requests.RequestException as exc:  # pragma: no cover - network
            LOGGER.warning("Request failed for %s: %s", url, exc)
            return None
        if resp.status_code == 404:
            return None
        if resp.status_code >= 500:
            LOGGER.warning("Server error %s for %s", resp.status_code, url)
            return None
        return resp

    # Lookup helpers ---------------------------------------------------

    def _lookup_gene_symbol(self, gene_id: str) -> str:
        """Look up the gene symbol for an Ensembl gene ID.

        Results are cached to avoid repeated requests.
        """
        if gene_id in self._symbol_cache:
            return self._symbol_cache[gene_id]
        url = f"{self.base_url}/lookup/id/{gene_id}"
        resp = self._request(url, [("content-type", "application/json")])
        symbol = ""
        if resp:
            try:
                symbol = resp.json().get("display_name", "")
            except ValueError:
                symbol = ""
        self._symbol_cache[gene_id] = symbol
        return symbol

    def _lookup_uniprot(self, gene_id: str) -> str:
        """Look up the UniProt/Swiss-Prot accession for an Ensembl gene ID.

        Results are cached to avoid repeated requests.
        """
        if gene_id in self._uniprot_cache:
            return self._uniprot_cache[gene_id]
        url = f"{self.base_url}/xrefs/id/{gene_id}"
        params = [
            ("content-type", "application/json"),
            ("external_db", "UniProtKB/Swiss-Prot"),
        ]
        resp = self._request(url, params)
        acc = ""
        if resp:
            try:
                data = resp.json()
            except ValueError:
                data = []
            for item in data:
                if item.get("dbname", "") in {
                    "UniProtKB/Swiss-Prot",
                    "Uniprot/SWISSPROT",
                }:
                    acc = item.get("primary_id", "")
                    break
        self._uniprot_cache[gene_id] = acc
        return acc

    # ------------------------------------------------------------------

    def get_orthologs(
        self, ensembl_gene_id: str, target_species: List[str]
    ) -> List[Ortholog]:
        """Retrieve orthologs for ``ensembl_gene_id``.

        Parameters
        ----------
        ensembl_gene_id:
            Ensembl gene identifier for the query species.
        target_species:
            List of species names to query. Aliases like ``mouse`` are mapped to
            Ensembl internal names.
        """

        species = _species_from_gene(ensembl_gene_id)
        if not species:
            LOGGER.warning("Unknown species for gene %s", ensembl_gene_id)
            return []

        params: List[tuple[str, str]] = [
            ("content-type", "application/json"),
            ("type", "orthologues"),
        ]
        for sp in target_species:
            params.append(("target_species", _map_species(sp)))
        url = f"{self.base_url}/homology/id/{species}/{ensembl_gene_id}"
        resp = self._request(url, params)
        if not resp:
            return []
        try:
            data = resp.json()
        except ValueError:  # pragma: no cover - API guarantees JSON
            return []

        # The Ensembl API returns a "data" list that may be empty when no
        # orthologs are available.  Guard against this situation to avoid
        # ``IndexError`` when attempting to access the first element.
        homologies_container: Dict[str, Any] = {}
        records = data.get("data")
        if isinstance(records, list):
            for record in records:
                if isinstance(record, dict):
                    homologies_container = record
                    break
        if not homologies_container:
            LOGGER.debug(
                "No orthologs returned for %s (records: %s)",
                ensembl_gene_id,
                records,
            )
            return []

        homs = homologies_container.get("homologies", [])
        orthologs: List[Ortholog] = []
        for hom in homs:
            tgt = hom.get("target", {})
            gene_id = tgt.get("id", "")
            species_t = tgt.get("species", "")
            orth = Ortholog(
                target_species=species_t,
                target_gene_symbol=self._lookup_gene_symbol(gene_id),
                target_ensembl_gene_id=gene_id,
                target_uniprot_id=self._lookup_uniprot(gene_id) or None,
                homology_type=hom.get("type"),
                perc_id=tgt.get("perc_id"),
                perc_pos=tgt.get("perc_pos"),
                is_high_confidence=hom.get("is_high_confidence"),
                dn=(hom.get("dn_ds") or {}).get("dn"),
                ds=(hom.get("dn_ds") or {}).get("ds"),
                source_db="Ensembl",
            )
            orthologs.append(orth)
        orthologs.sort(key=lambda o: (o.target_species, o.target_gene_symbol))
        return orthologs


# ---------------------------------------------------------------------------
# OMA client (minimal placeholder)


@dataclass
class OmaClient:
    """Very small wrapper around the OMA REST API.

    The current implementation serves as a fallback and returns an empty list
    when no data can be retrieved.  The interface mirrors that of
    :class:`EnsemblHomologyClient` to allow easy substitution."""

    base_url: str
    network: NetworkConfig
    rate_limit: RateLimitConfig
    session: requests.Session = field(default_factory=requests.Session)

    def get_orthologs_by_uniprot(self, uniprot_id: str) -> List[Ortholog]:
        """Return orthologs for ``uniprot_id``.

        The OMA API is not queried in detail here; the function returns an empty
        list if the request fails or the endpoint is not available."""

        url = f"{self.base_url}/orthologs/{uniprot_id}/"
        try:
            resp = self.session.get(
                url,
                timeout=self.network.timeout_sec,
                headers={"Accept": "application/json"},
            )
        except requests.RequestException:  # pragma: no cover - network failure
            return []
        if resp.status_code != 200:
            return []
        try:
            data = resp.json()
        except ValueError:
            return []
        orthologs: List[Ortholog] = []
        for item in data.get("orthologs", []):  # pragma: no cover - unknown schema
            orthologs.append(
                Ortholog(
                    target_species=item.get("species", ""),
                    target_gene_symbol=item.get("gene", ""),
                    target_ensembl_gene_id=item.get("ensembl_gene", ""),
                    target_uniprot_id=item.get("uniprot_id"),
                    homology_type=item.get("type"),
                    perc_id=item.get("perc_id"),
                    perc_pos=item.get("perc_pos"),
                    source_db="OMA",
                )
            )
        orthologs.sort(key=lambda o: (o.target_species, o.target_gene_symbol))
        return orthologs
