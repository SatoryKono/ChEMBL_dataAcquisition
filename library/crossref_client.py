"""Client for the Crossref API."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Sequence
from urllib.parse import quote

import requests

from .http_client import HttpClient

LOGGER = logging.getLogger(__name__)

API_URL = "https://api.crossref.org/works/{doi}"


def _normalise_subject(subject: Any) -> List[str]:
    if isinstance(subject, list):
        return [str(item).strip() for item in subject if str(item).strip()]
    if isinstance(subject, str) and subject.strip():
        return [subject.strip()]
    return []


@dataclass
class CrossrefRecord:
    """Container for a Crossref work."""

    doi: str
    type: str | None
    subtype: str | None
    title: str | None
    subtitle: str | None
    subject: List[str]
    error: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of the record."""

        return {
            "crossref.DOI": self.doi,
            "crossref.Type": self.type,
            "crossref.Subtype": self.subtype,
            "crossref.Title": self.title,
            "crossref.Subtitle": self.subtitle,
            "crossref.Subject": "|".join(self.subject),
            "crossref.Error": self.error,
        }

    @classmethod
    def from_error(cls, doi: str, message: str) -> "CrossrefRecord":
        return cls(
            doi=doi,
            type=None,
            subtype=None,
            title=None,
            subtitle=None,
            subject=[],
            error=message,
        )


def fetch_crossref_records(
    dois: Sequence[str],
    *,
    client: HttpClient,
) -> List[CrossrefRecord]:
    """Fetch metadata from Crossref for a sequence of DOIs."""

    cleaned = [doi.strip() for doi in dois if doi and doi.strip()]
    if not cleaned:
        return []

    records: Dict[str, CrossrefRecord] = {}
    for doi in cleaned:
        url = API_URL.format(doi=quote(doi))
        LOGGER.debug("Requesting Crossref for DOI %s", doi)
        try:
            resp = client.request("get", url)
        except requests.HTTPError as exc:  # pragma: no cover
            status = exc.response.status_code if exc.response is not None else "N/A"
            msg = f"HTTP error {status}"
            LOGGER.warning("Crossref request failed for %s: %s", doi, msg)
            records[doi] = CrossrefRecord.from_error(doi, msg)
            continue
        except requests.RequestException as exc:  # pragma: no cover
            msg = f"Request error: {exc}"
            LOGGER.warning("Crossref request failed for %s: %s", doi, msg)
            records[doi] = CrossrefRecord.from_error(doi, msg)
            continue

        if resp.status_code >= 400:
            msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
            LOGGER.warning("Crossref returned error for %s: %s", doi, msg)
            records[doi] = CrossrefRecord.from_error(doi, msg)
            continue

        try:
            payload = resp.json()
        except ValueError as exc:
            msg = f"JSON decode error: {exc}"
            LOGGER.warning("Crossref JSON decode failed for %s: %s", doi, msg)
            records[doi] = CrossrefRecord.from_error(doi, msg)
            continue

        message = payload.get("message") if isinstance(payload, dict) else None
        if not isinstance(message, dict):
            msg = "Unexpected response payload"
            LOGGER.warning(
                "Crossref responded with %s for DOI %s", type(payload).__name__, doi
            )
            records[doi] = CrossrefRecord.from_error(doi, msg)
            continue

        titles = message.get("title") or []
        title = titles[0].strip() if titles and isinstance(titles[0], str) else None
        subtitles = message.get("subtitle") or []
        subtitle = (
            subtitles[0].strip()
            if subtitles and isinstance(subtitles[0], str)
            else None
        )
        subject = _normalise_subject(message.get("subject"))

        records[doi] = CrossrefRecord(
            doi=doi,
            type=message.get("type"),
            subtype=message.get("subtype"),
            title=title,
            subtitle=subtitle,
            subject=subject,
            error=None,
        )

    return [records[doi] for doi in cleaned]


__all__ = ["CrossrefRecord", "fetch_crossref_records"]
