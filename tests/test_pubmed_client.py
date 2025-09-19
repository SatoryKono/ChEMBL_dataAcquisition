"""Tests for the PubMed client helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Sequence

import requests_mock

# Ensure modules are importable when the test is executed in isolation
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from library import pubmed_client as pc  # type: ignore  # noqa: E402
from library.http_client import HttpClient  # type: ignore  # noqa: E402


def test_fetch_pubmed_records_parses_fields(
    pubmed_xml_factory: Callable[[Sequence[tuple[str, str, str | None]]], str],
) -> None:
    """Basic parsing should surface the expected bibliographic fields."""

    xml_payload = pubmed_xml_factory(
        [("1", "Title 1", "Journal Article"), ("2", "Title 2", "Review")]
    )
    with requests_mock.Mocker() as m:
        m.get(pc.API_URL, text=xml_payload)
        client = HttpClient(timeout=1.0, max_retries=1, rps=0)
        records = pc.fetch_pubmed_records(["1", "2"], client=client, batch_size=200)

    assert [r.pmid for r in records] == ["1", "2"]
    assert records[0].doi == "10.1/doi1"
    assert records[0].title == "Title 1"
    assert pc.classify_publication(records[0].publication_types) == "experimental"
    assert pc.classify_publication(records[1].publication_types) == "review"
    review_score, experimental_score, normalised = pc.score_publication_types(
        ["Review", "Meta Analysis"], source="pubmed"
    )
    assert review_score > experimental_score
    assert "review" in normalised


def test_fetch_pubmed_records_batches_requests(
    pubmed_xml_factory: Callable[[Sequence[tuple[str, str, str | None]]], str],
) -> None:
    """The downloader should honour the requested batch size for pagination."""

    xml_chunk_one = pubmed_xml_factory(
        [("1", "Title 1", "Journal Article"), ("2", "Title 2", "Review")]
    )
    xml_chunk_two = pubmed_xml_factory([("3", "Title 3", None)])

    def _match(expected: str):
        def _matcher(request: requests_mock.request._RequestObjectProxy) -> bool:  # type: ignore[attr-defined]
            ids = request.qs.get("id", [])
            return ids == [expected]

        return _matcher

    with requests_mock.Mocker() as m:
        m.get(pc.API_URL, text=xml_chunk_one, additional_matcher=_match("1,2"))
        m.get(pc.API_URL, text=xml_chunk_two, additional_matcher=_match("3"))
        client = HttpClient(timeout=1.0, max_retries=1, rps=0)
        records = pc.fetch_pubmed_records(["1", "2", "3"], client=client, batch_size=2)
        history = list(m.request_history)

    assert [req.qs["id"][0] for req in history] == ["1,2", "3"]
    assert [record.pmid for record in records] == ["1", "2", "3"]


def test_fetch_pubmed_records_reports_http_error() -> None:
    """HTTP status errors should be converted into error records."""

    with requests_mock.Mocker() as m:
        m.get(pc.API_URL, status_code=500, text="boom")
        client = HttpClient(timeout=1.0, max_retries=1, rps=0)
        records = pc.fetch_pubmed_records(["99"], client=client, batch_size=1)

    assert records[0].error is not None
    assert records[0].error == "HTTP error 500"


def test_fetch_pubmed_records_marks_missing_pmids(
    pubmed_xml_factory: Callable[[Sequence[tuple[str, str, str | None]]], str],
) -> None:
    """Missing identifiers must surface descriptive error messages."""

    xml_payload = pubmed_xml_factory([("1", "Title 1", None)])
    with requests_mock.Mocker() as m:
        m.get(pc.API_URL, text=xml_payload)
        client = HttpClient(timeout=1.0, max_retries=1, rps=0)
        records = pc.fetch_pubmed_records(["1", "2"], client=client, batch_size=5)

    assert records[0].error is None
    assert records[1].error == "PMID not returned by PubMed"
