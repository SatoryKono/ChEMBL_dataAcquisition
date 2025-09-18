"""Client for the Crossref API."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib.parse import quote

import requests

from .http_client import HttpClient

LOGGER = logging.getLogger(__name__)

API_URL = "https://api.crossref.org/works/{doi}"


def _clean_string(value: Any) -> str | None:
    if isinstance(value, str):
        candidate = value.strip()
        if candidate:
            return candidate
    return None


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    """Return ``values`` without duplicates while preserving the input order."""

    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _coerce_sequence(value: Any) -> List[Any]:
    """Return ``value`` as a list while preserving the original ordering."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _normalise_strings(value: Any) -> List[str]:
    """Extract cleaned string values from ``value`` preserving order."""

    strings: List[str] = []
    for item in _coerce_sequence(value):
        if isinstance(item, str):
            candidate = _clean_string(item)
            if candidate:
                strings.append(candidate)
    return strings


def _normalise_subject(subject: Any) -> List[str]:
    values: List[str] = []
    for item in _coerce_sequence(subject):
        if isinstance(item, Mapping):
            for key in ("name", "value"):
                name = _clean_string(item.get(key))
                if name:
                    values.append(name)
                    break
            continue
        text = _clean_string(item)
        if text:
            values.append(text)
    return _dedupe_preserve_order(values)


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

        titles = _normalise_strings(message.get("title"))
        title = titles[0] if titles else None

        subtitle_values = _normalise_strings(message.get("subtitle"))
        subtitle = "|".join(subtitle_values) if subtitle_values else None

        subtype_values = _normalise_strings(message.get("subtype"))
        subtype = subtype_values[0] if subtype_values else None

        subject = _normalise_subject(message.get("subject"))

        records[doi] = CrossrefRecord(
            doi=doi,
            type=_clean_string(message.get("type")),
            subtype=subtype,
            title=title,
            subtitle=subtitle,
            subject=subject,
            error=None,
        )

    return [records[doi] for doi in cleaned]


__all__ = ["CrossrefRecord", "fetch_crossref_records"]
