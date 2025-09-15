"""Client for the Crossref API."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Sequence

from .http_client import HttpClient

LOGGER = logging.getLogger(__name__)

API_URL = "https://api.crossref.org/works/{doi}"


@dataclass
class CrossrefRecord:
    """Container for a Crossref work."""

    doi: str
    type: str | None
    subtype: str | None
    title: str | None
    subject: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of the record."""

        return {
            "crossref.DOI": self.doi,
            "crossref.Type": self.type,
            "crossref.Subtype": self.subtype,
            "crossref.Title": self.title,
            "crossref.Subject": "|".join(self.subject),
        }


def fetch_crossref_records(
    dois: Sequence[str], *, client: HttpClient
) -> List[CrossrefRecord]:
    """Fetch metadata from Crossref for a sequence of DOIs."""

    records: List[CrossrefRecord] = []
    for doi in dois:
        url = API_URL.format(doi=doi)
        LOGGER.debug("Requesting Crossref for DOI %s", doi)
        resp = client.request("get", url)
        item = resp.json().get("message", {})
        title_list = item.get("title") or []
        title = title_list[0] if title_list else None
        subject = item.get("subject") or []
        records.append(
            CrossrefRecord(
                doi=doi,
                type=item.get("type"),
                subtype=item.get("subtype"),
                title=title,
                subject=subject,
            )
        )
    return records


__all__ = ["CrossrefRecord", "fetch_crossref_records"]
