"""Client for fetching PubMed records.

This module provides a small convenience wrapper around the NCBI E-utilities
API.  Identifiers are downloaded in batches and a minimal subset of fields is
extracted into dictionaries ready for further processing.

The implementation intentionally covers only a handful of metadata fields to
keep the example concise.  It can be extended in the future to expose more of
the rich PubMed schema.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable, List, Dict, Any, Sequence
import xml.etree.ElementTree as ET

from .http_client import HttpClient

LOGGER = logging.getLogger(__name__)

API_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


@dataclass
class PubMedRecord:
    """Container for a parsed PubMed article."""

    pmid: str
    doi: str | None
    title: str | None
    abstract: str | None
    journal: str | None
    publication_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of the record."""

        return {
            "PubMed.PMID": self.pmid,
            "PubMed.DOI": self.doi,
            "PubMed.ArticleTitle": self.title,
            "PubMed.Abstract": self.abstract,
            "PubMed.JournalTitle": self.journal,
            "PubMed.PublicationType": "|".join(self.publication_types),
        }


def _parse_article(article: ET.Element) -> PubMedRecord:
    """Parse a single ``PubmedArticle`` element."""

    pmid = article.findtext("MedlineCitation/PMID", default="")
    doi = None
    for id_node in article.findall("PubmedData/ArticleIdList/ArticleId"):
        if id_node.attrib.get("IdType") == "doi":
            doi = id_node.text
            break
    title = article.findtext("MedlineCitation/Article/ArticleTitle", default=None)
    abstract = article.findtext(
        "MedlineCitation/Article/Abstract/AbstractText", default=None
    )
    journal = article.findtext("MedlineCitation/Article/Journal/Title", default=None)
    publication_types = [
        node.text or ""
        for node in article.findall(
            "MedlineCitation/Article/PublicationTypeList/PublicationType"
        )
    ]
    return PubMedRecord(
        pmid=pmid,
        doi=doi,
        title=title,
        abstract=abstract,
        journal=journal,
        publication_types=publication_types,
    )


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
        Parsed records in the same order as the input identifiers.
    """

    records: List[PubMedRecord] = []
    ids = list(pmids)
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        params = {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"}
        LOGGER.debug("Requesting %d PubMed IDs", len(batch))
        resp = client.request("get", API_URL, params=params)
        root = ET.fromstring(resp.text)
        for article in root.findall("PubmedArticle"):
            records.append(_parse_article(article))
    return records


def classify_publication(publication_types: Iterable[str]) -> str:
    """Classify a publication into broad categories.

    The heuristics are intentionally simple: if any type mentions ``review`` the
    record is labelled as ``"review"``; if types indicate a typical experimental
    article the label ``"experimental"`` is used.  Otherwise ``"unknown"`` is
    returned.
    """

    lowered = {t.strip().lower() for t in publication_types}
    if any("review" in t for t in lowered):
        return "review"
    experimental_markers = {"journal article", "clinical trial", "case report"}
    if lowered & experimental_markers:
        return "experimental"
    return "unknown"


__all__ = ["PubMedRecord", "fetch_pubmed_records", "classify_publication"]
