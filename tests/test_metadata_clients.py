"""Tests for Semantic Scholar, OpenAlex and Crossref clients."""

from __future__ import annotations

import sys
from pathlib import Path

import requests_mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from library.http_client import HttpClient  # type: ignore  # noqa: E402
from library.semantic_scholar_client import (  # type: ignore  # noqa: E402
    API_URL as SS_URL,
    fetch_semantic_scholar_records,
)
from library.openalex_client import (  # type: ignore  # noqa: E402
    API_URL as OA_URL,
    fetch_openalex_records,
)
from library.crossref_client import (  # type: ignore  # noqa: E402
    API_URL as CR_URL,
    fetch_crossref_records,
)


def test_semantic_scholar_parses_fields():
    sample = [
        {
            "paperId": "S1",
            "externalIds": {"PMID": "1", "DOI": "10.1/doi1"},
            "publicationTypes": ["JournalArticle"],
            "venue": "Venue1",
        },
        {
            "paperId": "S2",
            "externalIds": {"PMID": "2"},
            "publicationTypes": ["Review"],
            "venue": "Venue2",
        },
    ]
    with requests_mock.Mocker() as m:
        m.post(SS_URL, json=sample)
        client = HttpClient(timeout=1.0, max_retries=1, rps=0)
        recs = fetch_semantic_scholar_records(["1", "2"], client=client)
    assert recs[0].doi == "10.1/doi1"
    assert recs[1].publication_types == ["Review"]


def test_openalex_parses_fields():
    with requests_mock.Mocker() as m:
        m.get(
            OA_URL.format(pmid="1"),
            json={
                "id": "https://openalex.org/W1",
                "doi": "https://doi.org/10.1/doi1",
                "publication_types": ["journal-article"],
                "type_crossref": "journal-article",
                "genre": "journal-article",
                "host_venue": {"display_name": "Journal"},
            },
        )
        client = HttpClient(timeout=1.0, max_retries=1, rps=0)
        recs = fetch_openalex_records(["1"], client=client)
    assert recs[0].doi == "10.1/doi1"
    assert recs[0].venue == "Journal"


def test_crossref_parses_fields():
    doi = "10.1/doi1"
    with requests_mock.Mocker() as m:
        m.get(
            CR_URL.format(doi=doi),
            json={
                "message": {
                    "type": "journal-article",
                    "subtype": "",
                    "title": ["Title"],
                    "subject": ["Biology"],
                }
            },
        )
        client = HttpClient(timeout=1.0, max_retries=1, rps=0)
        recs = fetch_crossref_records([doi], client=client)
    assert recs[0].title == "Title"
    assert recs[0].subject == ["Biology"]
