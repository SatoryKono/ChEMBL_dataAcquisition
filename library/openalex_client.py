"""Client for the OpenAlex API."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Sequence

from .http_client import HttpClient

LOGGER = logging.getLogger(__name__)

API_URL = "https://api.openalex.org/works/pmid:{pmid}"


@dataclass
class OpenAlexRecord:
    """Container for an OpenAlex work."""

    pmid: str
    doi: str | None
    publication_types: List[str]
    type_crossref: str | None
    genre: str | None
    venue: str | None
    work_id: str | None

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of the record."""

        return {
            "OpenAlex.PMID": self.pmid,
            "OpenAlex.DOI": self.doi,
            "OpenAlex.PublicationTypes": "|".join(self.publication_types),
            "OpenAlex.TypeCrossref": self.type_crossref,
            "OpenAlex.Genre": self.genre,
            "OpenAlex.Venue": self.venue,
            "OpenAlex.Id": self.work_id,
        }


def fetch_openalex_records(
    pmids: Sequence[str], *, client: HttpClient
) -> List[OpenAlexRecord]:
    """Fetch metadata from OpenAlex for a sequence of PMIDs."""

    records: List[OpenAlexRecord] = []
    for pmid in pmids:
        url = API_URL.format(pmid=pmid)
        LOGGER.debug("Requesting OpenAlex for PMID %s", pmid)
        resp = client.request("get", url)
        item = resp.json()
        doi = item.get("doi")
        if isinstance(doi, str) and doi.startswith("https://doi.org/"):
            doi = doi.replace("https://doi.org/", "")
        publication_types = item.get("publication_types") or []
        type_crossref = item.get("type_crossref")
        genre = item.get("genre")
        venue = None
        host = item.get("host_venue")
        if isinstance(host, dict):
            venue = host.get("display_name")
        work_id = item.get("id")
        records.append(
            OpenAlexRecord(
                pmid=pmid,
                doi=doi,
                publication_types=publication_types,
                type_crossref=type_crossref,
                genre=genre,
                venue=venue,
                work_id=work_id,
            )
        )
    return records


__all__ = ["OpenAlexRecord", "fetch_openalex_records"]
