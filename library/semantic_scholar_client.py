"""Client for the Semantic Scholar API with defensive parsing."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Dict, List, Sequence

import requests

from .http_client import HttpClient

LOGGER = logging.getLogger(__name__)

API_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
DEFAULT_FIELDS = ["externalIds", "publicationTypes", "venue", "paperId"]


def _chunked(seq: Sequence[str], size: int) -> List[List[str]]:
    return [list(seq[i : i + size]) for i in range(0, len(seq), size)]


@dataclass
class SemanticScholarRecord:
    """Container for a Semantic Scholar response."""

    pmid: str
    doi: str | None
    publication_types: List[str]
    venue: str | None
    paper_id: str | None
    external_ids: Dict[str, Any]
    error: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of the record."""

        return {
            "scholar.PMID": self.pmid,
            "scholar.DOI": self.doi,
            "scholar.PublicationTypes": "|".join(self.publication_types),
            "scholar.Venue": self.venue,
            "scholar.SemanticScholarId": self.paper_id,
            "scholar.ExternalIds": (
                json.dumps(self.external_ids, separators=(",", ":"))
                if self.external_ids
                else None
            ),
            "scholar.Error": self.error,
        }

    @classmethod
    def from_error(cls, pmid: str, message: str) -> "SemanticScholarRecord":
        return cls(
            pmid=pmid,
            doi=None,
            publication_types=[],
            venue=None,
            paper_id=None,
            external_ids={},
            error=message,
        )


def _parse_item(raw: Any, fallback_pmid: str) -> SemanticScholarRecord:
    if not isinstance(raw, dict):
        return SemanticScholarRecord.from_error(
            fallback_pmid, "Unexpected payload type from Semantic Scholar"
        )

    if "error" in raw:
        return SemanticScholarRecord.from_error(fallback_pmid, str(raw["error"]))

    external_ids = raw.get("externalIds") or {}
    pmid = str(external_ids.get("PMID") or fallback_pmid)
    doi = external_ids.get("DOI")
    if isinstance(doi, str):
        doi = doi.strip() or None
    publication_types = [
        str(value).strip()
        for value in raw.get("publicationTypes") or []
        if str(value).strip()
    ]
    venue = raw.get("venue")
    if isinstance(venue, str):
        venue = venue.strip() or None
    paper_id = raw.get("paperId")
    if isinstance(paper_id, str):
        paper_id = paper_id.strip() or None

    return SemanticScholarRecord(
        pmid=pmid,
        doi=doi,
        publication_types=publication_types,
        venue=venue,
        paper_id=paper_id,
        external_ids={
            key: value for key, value in external_ids.items() if value is not None
        },
        error=None,
    )


def fetch_semantic_scholar_records(
    pmids: Sequence[str],
    *,
    client: HttpClient,
    chunk_size: int = 100,
) -> List[SemanticScholarRecord]:
    """Fetch metadata from Semantic Scholar.

    Parameters
    ----------
    pmids
        Sequence of PubMed identifiers.
    client
        HTTP client used for requests.
    chunk_size
        Maximum number of identifiers passed to the batch endpoint.

    Returns
    -------
    list of :class:`SemanticScholarRecord`
        Parsed records in the same order as input identifiers.
    """

    cleaned = [pid for pid in (p.strip() for p in pmids) if pid]
    if not cleaned:
        return []

    records: Dict[str, SemanticScholarRecord] = {}
    for chunk in _chunked(cleaned, chunk_size):
        payload = {
            "ids": [f"PMID:{p}" for p in chunk],
            "fields": ",".join(DEFAULT_FIELDS),
        }
        LOGGER.debug("Requesting %d Semantic Scholar IDs", len(chunk))
        try:
            resp = client.request("post", API_URL, json=payload)
        except requests.HTTPError as exc:  # pragma: no cover
            status = exc.response.status_code if exc.response is not None else "N/A"
            msg = f"HTTP error {status}"
            LOGGER.warning("Semantic Scholar batch failed: %s", msg)
            for pmid in chunk:
                records[pmid] = SemanticScholarRecord.from_error(pmid, msg)
            continue
        except requests.RequestException as exc:  # pragma: no cover
            msg = f"Request error: {exc}"
            LOGGER.warning("Semantic Scholar batch failed: %s", msg)
            for pmid in chunk:
                records[pmid] = SemanticScholarRecord.from_error(pmid, msg)
            continue

        if resp.status_code >= 400:
            msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
            LOGGER.warning("Semantic Scholar returned error status: %s", msg)
            for pmid in chunk:
                records[pmid] = SemanticScholarRecord.from_error(pmid, msg)
            continue

        try:
            data = resp.json()
        except ValueError as exc:
            msg = f"JSON decode error: {exc}"
            LOGGER.warning("Semantic Scholar JSON decode failed: %s", msg)
            for pmid in chunk:
                records[pmid] = SemanticScholarRecord.from_error(pmid, msg)
            continue

        if not isinstance(data, list):
            msg = "Unexpected response payload"
            LOGGER.warning("Semantic Scholar responded with %s", type(data).__name__)
            for pmid in chunk:
                records[pmid] = SemanticScholarRecord.from_error(pmid, msg)
            continue

        for fallback, item in zip(chunk, data):
            record = _parse_item(item, fallback)
            records[record.pmid] = record

        # Ensure that identifiers missing in the response are represented.
        for pmid in chunk:
            records.setdefault(
                pmid, SemanticScholarRecord.from_error(pmid, "PMID missing")
            )

    ordered: List[SemanticScholarRecord] = []
    for pmid in pmids:
        key = pmid.strip()
        if not key:
            continue
        ordered.append(
            records.get(
                key, SemanticScholarRecord.from_error(key, "PMID not requested")
            )
        )
    return ordered


__all__ = ["SemanticScholarRecord", "fetch_semantic_scholar_records"]
