"""Client for the OpenAlex API."""

from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import logging
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import requests

from .http_client import HttpClient

LOGGER = logging.getLogger(__name__)

API_URL = "https://api.openalex.org/works/pmid:{pmid}"


def _clean_string(value: Any) -> str | None:
    if isinstance(value, str):
        candidate = value.strip()
        if candidate:
            return candidate
    return None


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    return None


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _strip_html(value: str) -> str:
    """Return a compact representation of ``value`` with HTML markup removed."""

    if not value:
        return ""
    # Replace HTML tags with whitespace before normalising consecutive spaces.
    without_tags = re.sub(r"<[^>]+>", " ", unescape(value))
    compact = re.sub(r"\s+", " ", without_tags).strip()
    return compact[:200]


def _extract_text(value: Any) -> str | None:
    """Extract the first meaningful text snippet from ``value``."""

    if isinstance(value, str):
        candidate = value.strip()
        if candidate:
            return candidate
        return None
    if isinstance(value, Mapping):
        for key in ("message", "error", "detail", "description", "title", "reason"):
            text = _extract_text(value.get(key))
            if text:
                return text
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            text = _extract_text(item)
            if text:
                return text
    return None


def _format_error_message(response: requests.Response) -> str:
    """Return a human readable summary of an erroneous HTTP ``response``."""

    status = response.status_code
    raw_reason = response.reason
    reason = raw_reason or "Unknown error"

    message: str | None = None
    try:
        payload = response.json()
    except ValueError:
        payload = None
    else:
        message = _extract_text(payload)

    if not message:
        message = _strip_html(response.text)

    if status == 404:
        fallback = raw_reason or "Not Found"
        if not message or message.lower().startswith("not found"):
            message = fallback

    if not message:
        message = reason

    return f"HTTP {status}: {message}"


def _extract_publication_types(item: Mapping[str, Any]) -> List[str]:
    values: List[str] = []
    raw_types = item.get("publication_types")
    if isinstance(raw_types, list):
        values.extend(
            [entry for entry in (_clean_string(val) for val in raw_types) if entry]
        )
    else:
        single = _clean_string(raw_types)
        if single:
            values.append(single)

    fallback_candidates = [
        _clean_string(item.get("type")),
        _clean_string(item.get("type_crossref")),
    ]
    primary_location = _as_mapping(item.get("primary_location"))
    if primary_location:
        source = _as_mapping(primary_location.get("source"))
        if source:
            fallback_candidates.append(_clean_string(source.get("type")))

    for candidate in fallback_candidates:
        if candidate:
            values.append(candidate)

    return _dedupe_preserve_order(values)


def _extract_venue(item: Mapping[str, Any]) -> str | None:
    host = _as_mapping(item.get("host_venue"))
    if host:
        venue = _clean_string(host.get("display_name"))
        if venue:
            return venue

    primary_location = _as_mapping(item.get("primary_location"))
    if primary_location:
        venue = _clean_string(primary_location.get("display_name"))
        if venue:
            return venue
        source = _as_mapping(primary_location.get("source"))
        if source:
            venue = _clean_string(source.get("display_name"))
            if venue:
                return venue

    locations = item.get("locations")
    if isinstance(locations, list):
        for location in locations:
            location_map = _as_mapping(location)
            if not location_map:
                continue
            venue = _clean_string(location_map.get("display_name"))
            if venue:
                return venue
            source = _as_mapping(location_map.get("source"))
            if source:
                venue = _clean_string(source.get("display_name"))
                if venue:
                    return venue

    return None


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
            if exc.response is not None:
                msg = _format_error_message(exc.response)
            else:
                msg = f"HTTP error {getattr(exc.response, 'status_code', 'N/A')}"
            LOGGER.warning("OpenAlex request failed for PMID %s: %s", pmid, msg)
            records[pmid] = OpenAlexRecord.from_error(pmid, msg)
            continue
        except requests.RequestException as exc:  # pragma: no cover
            msg = f"Request error: {exc}"
            LOGGER.warning("OpenAlex request failed for PMID %s: %s", pmid, msg)
            records[pmid] = OpenAlexRecord.from_error(pmid, msg)
            continue

        if resp.status_code >= 400:
            msg = _format_error_message(resp)
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
        publication_types = _extract_publication_types(item)
        type_crossref = _clean_string(item.get("type_crossref"))
        genre = _clean_string(item.get("genre")) or _clean_string(item.get("type"))
        venue = _extract_venue(item)
        mesh_descriptors, mesh_qualifiers = _parse_mesh(item.get("mesh"))
        work_id = _clean_string(item.get("id"))

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
