"""Client for the Semantic Scholar API."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Sequence

from .http_client import HttpClient

LOGGER = logging.getLogger(__name__)

API_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"


@dataclass
class SemanticScholarRecord:
    """Container for a Semantic Scholar response."""

    pmid: str
    doi: str | None
    publication_types: List[str]
    venue: str | None
    paper_id: str | None

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of the record."""

        return {
            "scholar.PMID": self.pmid,
            "scholar.DOI": self.doi,
            "scholar.PublicationTypes": "|".join(self.publication_types),
            "scholar.Venue": self.venue,
            "scholar.SemanticScholarId": self.paper_id,
        }


def fetch_semantic_scholar_records(
    pmids: Sequence[str], *, client: HttpClient
) -> List[SemanticScholarRecord]:
    """Fetch metadata from Semantic Scholar.

    Parameters
    ----------
    pmids
        Sequence of PubMed identifiers.
    client
        HTTP client used for requests.

    Returns
    -------
    list of :class:`SemanticScholarRecord`
        Parsed records in the same order as input identifiers.
    """

    payload = {
        "ids": [f"PMID:{p}" for p in pmids],
        "fields": "externalIds,publicationTypes,venue,paperId",
    }
    LOGGER.debug("Requesting %d Semantic Scholar IDs", len(pmids))
    resp = client.request("post", API_URL, json=payload)
    data = resp.json()
    records: List[SemanticScholarRecord] = []
    for item in data:
        external = item.get("externalIds", {})
        pmid = external.get("PMID", "")
        doi = external.get("DOI")
        publication_types = item.get("publicationTypes") or []
        venue = item.get("venue")
        paper_id = item.get("paperId")
        records.append(
            SemanticScholarRecord(
                pmid=pmid,
                doi=doi,
                publication_types=publication_types,
                venue=venue,
                paper_id=paper_id,
            )
        )
    return records


__all__ = ["SemanticScholarRecord", "fetch_semantic_scholar_records"]
