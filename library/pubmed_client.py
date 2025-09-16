"""Robust client for downloading and parsing PubMed records.

The functions in this module wrap the NCBI E-utilities API to obtain PubMed
metadata in reproducible batches. XML payloads are converted into structured
dataclasses containing the most commonly used bibliographic fields as well as
MeSH descriptors, chemical annotations and completion/revision timestamps.

The parser is intentionally defensive: malformed responses, missing records or
HTTP failures are converted into :class:`PubMedRecord` instances with the
``error`` attribute populated. Callers can therefore compose the records
without guarding each network call explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable, List, Dict, Any, Sequence, Tuple
import xml.etree.ElementTree as ET

import requests

from .http_client import HttpClient

LOGGER = logging.getLogger(__name__)

API_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _concat_text(nodes: Iterable[ET.Element]) -> str:
    """Join the text content of ``nodes`` with whitespace separators."""

    parts: List[str] = []
    for node in nodes:
        text = (node.text or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def _safe_text(node: ET.Element | None) -> str | None:
    if node is None:
        return None
    text = (node.text or "").strip()
    return text or None


def _extract_date(node: ET.Element | None) -> Tuple[str | None, str | None, str | None]:
    if node is None:
        return (None, None, None)
    year = _safe_text(node.find("Year"))
    month = _safe_text(node.find("Month"))
    day = _safe_text(node.find("Day"))
    return year, month, day


@dataclass
class PubMedRecord:
    """Container for a parsed PubMed article."""

    pmid: str
    doi: str | None
    title: str | None
    abstract: str | None
    journal: str | None
    journal_abbrev: str | None
    volume: str | None
    issue: str | None
    start_page: str | None
    end_page: str | None
    issn: str | None
    publication_types: List[str]
    mesh_descriptors: List[str]
    mesh_qualifiers: List[str]
    chemical_list: List[str]
    year_completed: str | None
    month_completed: str | None
    day_completed: str | None
    year_revised: str | None
    month_revised: str | None
    day_revised: str | None
    error: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of the record."""

        return {
            "PubMed.PMID": self.pmid,
            "PubMed.DOI": self.doi,
            "PubMed.ArticleTitle": self.title,
            "PubMed.Abstract": self.abstract,
            "PubMed.JournalTitle": self.journal,
            "PubMed.JournalISOAbbrev": self.journal_abbrev,
            "PubMed.Volume": self.volume,
            "PubMed.Issue": self.issue,
            "PubMed.StartPage": self.start_page,
            "PubMed.EndPage": self.end_page,
            "PubMed.ISSN": self.issn,
            "PubMed.PublicationType": "|".join(self.publication_types),
            "PubMed.MeSH_Descriptors": "|".join(self.mesh_descriptors),
            "PubMed.MeSH_Qualifiers": "|".join(self.mesh_qualifiers),
            "PubMed.ChemicalList": "|".join(self.chemical_list),
            "PubMed.YearCompleted": self.year_completed,
            "PubMed.MonthCompleted": self.month_completed,
            "PubMed.DayCompleted": self.day_completed,
            "PubMed.YearRevised": self.year_revised,
            "PubMed.MonthRevised": self.month_revised,
            "PubMed.DayRevised": self.day_revised,
            "PubMed.Error": self.error,
        }

    @classmethod
    def from_error(cls, pmid: str, error: str) -> "PubMedRecord":
        """Create an error placeholder for ``pmid``."""

        return cls(
            pmid=pmid,
            doi=None,
            title=None,
            abstract=None,
            journal=None,
            journal_abbrev=None,
            volume=None,
            issue=None,
            start_page=None,
            end_page=None,
            issn=None,
            publication_types=[],
            mesh_descriptors=[],
            mesh_qualifiers=[],
            chemical_list=[],
            year_completed=None,
            month_completed=None,
            day_completed=None,
            year_revised=None,
            month_revised=None,
            day_revised=None,
            error=error,
        )


def _parse_article(article: ET.Element) -> PubMedRecord:
    """Parse a single ``PubmedArticle`` element."""

    pmid = article.findtext("MedlineCitation/PMID", default="").strip()
    doi = None
    for id_node in article.findall("PubmedData/ArticleIdList/ArticleId"):
        if id_node.attrib.get("IdType") == "doi":
            doi = (id_node.text or "").strip() or None
            if doi:
                break

    title = article.findtext("MedlineCitation/Article/ArticleTitle", default=None)
    if title is not None:
        title = title.strip()

    abstract_nodes = article.findall("MedlineCitation/Article/Abstract/AbstractText")
    abstract = _concat_text(abstract_nodes) if abstract_nodes else None

    journal = article.findtext("MedlineCitation/Article/Journal/Title", default=None)
    if journal is not None:
        journal = journal.strip()
    journal_abbrev = article.findtext(
        "MedlineCitation/Article/Journal/ISOAbbreviation", default=None
    )
    if journal_abbrev is not None:
        journal_abbrev = journal_abbrev.strip()

    journal_issue = article.find("MedlineCitation/Article/Journal/JournalIssue")
    volume = _safe_text(journal_issue.find("Volume") if journal_issue is not None else None)
    issue = _safe_text(journal_issue.find("Issue") if journal_issue is not None else None)
    start_page = article.findtext("MedlineCitation/Article/Pagination/StartPage")
    end_page = article.findtext("MedlineCitation/Article/Pagination/EndPage")
    start_page = start_page.strip() if isinstance(start_page, str) else None
    end_page = end_page.strip() if isinstance(end_page, str) else None
    issn = article.findtext("MedlineCitation/Article/Journal/ISSN")
    issn = issn.strip() if isinstance(issn, str) else None

    publication_types = [
        (node.text or "").strip()
        for node in article.findall(
            "MedlineCitation/Article/PublicationTypeList/PublicationType"
        )
        if (node.text or "").strip()
    ]

    mesh_descriptors: List[str] = []
    mesh_qualifiers: List[str] = []
    for heading in article.findall("MedlineCitation/MeshHeadingList/MeshHeading"):
        descriptor = heading.find("DescriptorName")
        descriptor_text = _safe_text(descriptor)
        if descriptor_text:
            mesh_descriptors.append(descriptor_text)
        qualifiers = heading.findall("QualifierName")
        for qualifier in qualifiers:
            qualifier_text = _safe_text(qualifier)
            if qualifier_text:
                mesh_qualifiers.append(qualifier_text)

    chemical_list = [
        name
        for name in (
            _safe_text(node.find("NameOfSubstance"))
            for node in article.findall("MedlineCitation/ChemicalList/Chemical")
        )
        if name
    ]

    year_completed, month_completed, day_completed = _extract_date(
        article.find("MedlineCitation/DateCompleted")
    )
    year_revised, month_revised, day_revised = _extract_date(
        article.find("MedlineCitation/DateRevised")
    )

    return PubMedRecord(
        pmid=pmid,
        doi=doi,
        title=title,
        abstract=abstract,
        journal=journal,
        journal_abbrev=journal_abbrev,
        volume=volume,
        issue=issue,
        start_page=start_page,
        end_page=end_page,
        issn=issn,
        publication_types=publication_types,
        mesh_descriptors=mesh_descriptors,
        mesh_qualifiers=mesh_qualifiers,
        chemical_list=chemical_list,
        year_completed=year_completed,
        month_completed=month_completed,
        day_completed=day_completed,
        year_revised=year_revised,
        month_revised=month_revised,
        day_revised=day_revised,
        error=None,
    )


def _chunk(seq: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def fetch_pubmed_records(
    pmids: Sequence[str],
    *,
    client: HttpClient,
    batch_size: int = 100,
) -> List[PubMedRecord]:
    """Fetch multiple PubMed records.

    Parameters
    ----------
    pmids:
        Sequence of PubMed identifiers to download.
    client:
        :class:`HttpClient` instance used for HTTP requests.
    batch_size:
        Maximum number of records to request in a single API call.

    Returns
    -------
    list of :class:`PubMedRecord`
        Parsed records in the same order as the input identifiers. Any failure
        is represented by a record whose ``error`` attribute is populated.
    """

    cleaned = [pid for pid in (p.strip() for p in pmids) if pid]
    if not cleaned:
        return []

    records: Dict[str, PubMedRecord] = {}
    for chunk in _chunk(cleaned, batch_size):
        params = {
            "db": "pubmed",
            "id": ",".join(chunk),
            "retmode": "xml",
            "rettype": "abstract",
        }
        LOGGER.debug("Requesting %d PubMed IDs", len(chunk))
        try:
            resp = client.request("get", API_URL, params=params)
        except requests.HTTPError as exc:  # pragma: no cover - exercised via status handling
            status = exc.response.status_code if exc.response is not None else "N/A"
            msg = f"HTTP error {status}"
            LOGGER.warning("PubMed batch failed: %s", msg)
            for pmid in chunk:
                records[pmid] = PubMedRecord.from_error(pmid, msg)
            continue
        except requests.RequestException as exc:  # pragma: no cover - network level
            msg = f"Request error: {exc}"
            LOGGER.warning("PubMed batch failed: %s", msg)
            for pmid in chunk:
                records[pmid] = PubMedRecord.from_error(pmid, msg)
            continue

        if resp.status_code >= 400:
            msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
            LOGGER.warning("PubMed batch returned error status: %s", msg)
            for pmid in chunk:
                records[pmid] = PubMedRecord.from_error(pmid, msg)
            continue

        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as exc:
            msg = f"XML parse error: {exc}"
            LOGGER.warning("Failed to parse PubMed XML: %s", msg)
            for pmid in chunk:
                records[pmid] = PubMedRecord.from_error(pmid, msg)
            continue

        seen: set[str] = set()
        for article in root.findall("PubmedArticle"):
            record = _parse_article(article)
            records[record.pmid] = record
            seen.add(record.pmid)

        missing = set(chunk) - seen
        for pmid in missing:
            msg = "PMID not returned by PubMed"
            records[pmid] = PubMedRecord.from_error(pmid, msg)

    ordered: List[PubMedRecord] = []
    for pmid in pmids:
        key = pmid.strip()
        if not key:
            continue
        record = records.get(key) or PubMedRecord.from_error(
            key, "PMID was not requested"
        )
        ordered.append(record)
    return ordered


_TYPE_SYNONYMS = {
    "journal article": "journal article",
    "journal-article": "journal article",
    "clinical trial": "clinical trial",
    "randomized controlled trial": "clinical trial",
    "randomised controlled trial": "clinical trial",
    "case report": "case report",
    "case reports": "case report",
    "review article": "review",
    "systematic review": "review",
    "systematic-review": "review",
    "meta analysis": "meta-analysis",
    "meta-analysis": "meta-analysis",
    "scoping review": "review",
    "literature review": "review",
    "mini review": "review",
    "research article": "journal article",
}

_REVIEW_TERMS = {"review", "meta-analysis"}
_EXPERIMENTAL_TERMS = {"journal article", "clinical trial", "case report"}
_SOURCE_WEIGHTS = {"pubmed": 3, "scholar": 2, "openalex": 1}


def _normalise_type(value: str) -> str | None:
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    cleaned = cleaned.replace("-", " ").replace("_", " ")
    cleaned = " ".join(cleaned.split())
    return _TYPE_SYNONYMS.get(cleaned, cleaned)


def score_publication_types(
    publication_types: Iterable[str], *, source: str | None = None
) -> tuple[int, int, List[str]]:
    """Return review/experimental scores and normalised terms for ``publication_types``."""

    weight = _SOURCE_WEIGHTS.get(source or "", 1)
    review_score = 0
    experimental_score = 0
    normalised: List[str] = []
    for raw in publication_types:
        norm = _normalise_type(raw or "")
        if not norm:
            continue
        normalised.append(norm)
        if norm in _REVIEW_TERMS:
            review_score += weight
        if norm in _EXPERIMENTAL_TERMS:
            experimental_score += weight
    return review_score, experimental_score, normalised


def classify_publication(
    publication_types: Iterable[str], *, source: str | None = None
) -> str:
    """Classify a publication into broad categories.

    Parameters
    ----------
    publication_types:
        Iterable of publication type strings from one or more sources.
    source:
        Optional origin of the types (``"pubmed"``, ``"scholar"`` or
        ``"openalex"``). When provided it is used to weight evidence during
        aggregation, otherwise all terms contribute equally.

    Returns
    -------
    str
        One of ``{"review", "experimental", "unknown"}``.
    """

    review_score, experimental_score, _ = score_publication_types(
        publication_types, source=source
    )
    if review_score and review_score >= experimental_score:
        return "review"
    if experimental_score:
        return "experimental"
    return "unknown"


__all__ = [
    "PubMedRecord",
    "fetch_pubmed_records",
    "classify_publication",
    "score_publication_types",
]
