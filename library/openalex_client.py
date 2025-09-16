"""Client for the OpenAlex API."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Sequence

import requests

from .http_client import HttpClient

LOGGER = logging.getLogger(__name__)

API_URL = "https://api.openalex.org/works/pmid:{pmid}"


def _normalise_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = doi.strip()
    if doi.startswith("https://doi.org/"):
        return doi.replace("https://doi.org/", "")
    if doi.startswith("http://doi.org/"):
        return doi.replace("http://doi.org/", "")
    return doi or None


@dataclass
class OpenAlexRecord:
    """Container for an OpenAlex work."""

    pmid: str
    doi: str | None
    publication_types: List[str]
    type_crossref: str | None
    genre: str | None
    venue: str | None
    mesh_descriptors: List[str]
    mesh_qualifiers: List[str]
    work_id: str | None
    error: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of the record."""

        return {
            "OpenAlex.PMID": self.pmid,
            "OpenAlex.DOI": self.doi,
            "OpenAlex.PublicationTypes": "|".join(self.publication_types),
            "OpenAlex.TypeCrossref": self.type_crossref,
            "OpenAlex.Genre": self.genre,
            "OpenAlex.Venue": self.venue,
            "OpenAlex.MeshDescriptors": "|".join(self.mesh_descriptors),
            "OpenAlex.MeshQualifiers": "|".join(self.mesh_qualifiers),
            "OpenAlex.Id": self.work_id,
            "OpenAlex.Error": self.error,
        }

    @classmethod
    def from_error(cls, pmid: str, message: str) -> "OpenAlexRecord":
        return cls(
            pmid=pmid,
            doi=None,
            publication_types=[],
            type_crossref=None,
            genre=None,
            venue=None,
            mesh_descriptors=[],
            mesh_qualifiers=[],
            work_id=None,
            error=message,
        )


def _parse_mesh(raw: Any) -> tuple[List[str], List[str]]:
    descriptors: List[str] = []
    qualifiers: List[str] = []
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            descriptor = entry.get("descriptor_name")
            qualifier_list = entry.get("qualifier_name")
            if isinstance(descriptor, str) and descriptor.strip():
                descriptors.append(descriptor.strip())
            if isinstance(qualifier_list, list):
                for qual in qualifier_list:
                    if isinstance(qual, str) and qual.strip():
                        qualifiers.append(qual.strip())
    return descriptors, qualifiers


def fetch_openalex_records(
    pmids: Sequence[str],
    *,
    client: HttpClient,
) -> List[OpenAlexRecord]:
    """Fetch metadata from OpenAlex for a sequence of PMIDs."""

    cleaned = [pid for pid in (p.strip() for p in pmids) if pid]
    if not cleaned:
        return []

    records: Dict[str, OpenAlexRecord] = {}
    for pmid in cleaned:
        url = API_URL.format(pmid=pmid)
        LOGGER.debug("Requesting OpenAlex for PMID %s", pmid)
        try:
            resp = client.request("get", url)
        except requests.HTTPError as exc:  # pragma: no cover
            status = exc.response.status_code if exc.response is not None else "N/A"
            msg = f"HTTP error {status}"
            LOGGER.warning("OpenAlex request failed for PMID %s: %s", pmid, msg)
            records[pmid] = OpenAlexRecord.from_error(pmid, msg)
            continue
        except requests.RequestException as exc:  # pragma: no cover
            msg = f"Request error: {exc}"
            LOGGER.warning("OpenAlex request failed for PMID %s: %s", pmid, msg)
            records[pmid] = OpenAlexRecord.from_error(pmid, msg)
            continue

        if resp.status_code >= 400:
            msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
            LOGGER.warning("OpenAlex returned error for PMID %s: %s", pmid, msg)
            records[pmid] = OpenAlexRecord.from_error(pmid, msg)
            continue

        try:
            item = resp.json()
        except ValueError as exc:
            msg = f"JSON decode error: {exc}"
            LOGGER.warning("OpenAlex JSON decode failed for PMID %s: %s", pmid, msg)
            records[pmid] = OpenAlexRecord.from_error(pmid, msg)
            continue

        if not isinstance(item, dict):
            msg = "Unexpected response payload"
            LOGGER.warning(
                "OpenAlex responded with %s for PMID %s", type(item).__name__, pmid
            )
            records[pmid] = OpenAlexRecord.from_error(pmid, msg)
            continue

        doi = _normalise_doi(item.get("doi"))
        publication_types = [
            str(value).strip()
            for value in item.get("publication_types") or []
            if str(value).strip()
        ]
        type_crossref = item.get("type_crossref")
        if isinstance(type_crossref, str):
            type_crossref = type_crossref.strip() or None
        genre = item.get("genre")
        if isinstance(genre, str):
            genre = genre.strip() or None
        venue = None
        host = item.get("host_venue")
        if isinstance(host, dict):
            display_name = host.get("display_name")
            if isinstance(display_name, str):
                venue = display_name.strip() or None
        mesh_descriptors, mesh_qualifiers = _parse_mesh(item.get("mesh"))
        work_id = item.get("id")
        if isinstance(work_id, str):
            work_id = work_id.strip() or None

        records[pmid] = OpenAlexRecord(
            pmid=pmid,
            doi=doi,
            publication_types=publication_types,
            type_crossref=type_crossref,
            genre=genre,
            venue=venue,
            mesh_descriptors=mesh_descriptors,
            mesh_qualifiers=mesh_qualifiers,
            work_id=work_id,
            error=None,
        )

    ordered: List[OpenAlexRecord] = []
    for pmid in pmids:
        key = pmid.strip()
        if not key:
            continue
        ordered.append(
            records.get(key, OpenAlexRecord.from_error(key, "PMID not requested"))
        )
    return ordered


__all__ = ["OpenAlexRecord", "fetch_openalex_records"]
