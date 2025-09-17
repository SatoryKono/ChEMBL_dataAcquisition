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
    DEFAULT_FIELDS,
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
            "externalIds": {"PubMed": "1", "DOI": "10.1/doi1", "CorpusId": 42},
            "publicationTypes": ["JournalArticle"],
            "venue": "Venue1",
        },
        {
            "paperId": "S2",
            "externalIds": {"PubMed": "2"},
            "publicationTypes": ["Review"],
            "venue": "Venue2",
        },
    ]
    with requests_mock.Mocker() as m:
        m.post(SS_URL, json=sample)
        client = HttpClient(timeout=1.0, max_retries=1, rps=0)
        recs = fetch_semantic_scholar_records(["1", "2"], client=client)
        assert m.last_request
        fields_param = m.last_request.qs.get("fields")
        assert fields_param is not None
        assert fields_param[0].lower() == ",".join(DEFAULT_FIELDS).lower()
    assert recs[0].doi == "10.1/doi1"
    assert recs[1].publication_types == ["Review"]
    assert recs[0].external_ids.get("CorpusId") == 42


def test_semantic_scholar_handles_string_error():
    with requests_mock.Mocker() as m:
        m.post(SS_URL, json="Bad request")
        client = HttpClient(timeout=1.0, max_retries=1, rps=0)
        recs = fetch_semantic_scholar_records(["1"], client=client)
    assert recs[0].error is not None


def test_openalex_parses_fields():
    with requests_mock.Mocker() as m:
        m.get(
            OA_URL.format(pmid="1"),
            json={
                "id": "https://openalex.org/W1",
                "doi": "https://doi.org/10.1/doi1",
                "type": "article",
                "type_crossref": "journal-article",
                "primary_location": {
                    "source": {"display_name": "Journal"},
                },
            },
        )
        client = HttpClient(timeout=1.0, max_retries=1, rps=0)
        recs = fetch_openalex_records(["1"], client=client)
    assert recs[0].doi == "10.1/doi1"
    assert recs[0].publication_types == ["article", "journal-article"]
    assert recs[0].genre == "article"
    assert recs[0].venue == "Journal"


def test_openalex_handles_html_error() -> None:
    """HTML error responses should be converted into readable messages."""

    html_body = (
        """<!doctype html><html><title>Not Found</title><body>Not Found</body></html>"""
    )
    with requests_mock.Mocker() as m:
        m.get(OA_URL.format(pmid="2"), status_code=404, text=html_body)
        client = HttpClient(timeout=1.0, max_retries=1, rps=0)
        recs = fetch_openalex_records(["2"], client=client)
    assert recs[0].error == "HTTP 404: Not Found"


def test_openalex_prefers_json_error_message() -> None:
    """JSON error payloads should surface the provided message."""

    with requests_mock.Mocker() as m:
        m.get(
            OA_URL.format(pmid="3"),
            status_code=429,
            json={"message": "Rate limit exceeded"},
            reason="Too Many Requests",
        )
        client = HttpClient(timeout=1.0, max_retries=1, rps=0)
        recs = fetch_openalex_records(["3"], client=client)
    assert recs[0].error == "HTTP 429: Rate limit exceeded"


def test_crossref_parses_fields():
    doi = "10.1/doi1"
    with requests_mock.Mocker() as m:
        m.get(
            CR_URL.format(doi=doi),
            json={
                "message": {
                    "type": "journal-article",
                    "subtype": " clinical-trial ",
                    "title": ["Title"],
                    "subject": [
                        {"name": "Biology"},
                        " Chemistry ",
                    ],
                }
            },
        )
        client = HttpClient(timeout=1.0, max_retries=1, rps=0)
        recs = fetch_crossref_records([doi], client=client)
    assert recs[0].title == "Title"
    assert recs[0].subtype == "clinical-trial"
    assert recs[0].subject == ["Biology", "Chemistry"]
