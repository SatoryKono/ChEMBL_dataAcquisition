"""Tests for the ChEMBL document client."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
import requests  # type: ignore[import-untyped]
import requests_mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from library.http_client import HttpClient  # type: ignore  # noqa: E402
from library.chembl_client import (  # type: ignore  # noqa: E402
    ApiCfg,
    ChemblBatchFetchError,
    ChemblClient,
    get_documents,
)


def test_get_documents_parses_response() -> None:
    cfg = ApiCfg()
    base = f"{cfg.chembl_base.rstrip('/')}/document.json?format=json"
    sample = {
        "documents": [
            {
                "document_chembl_id": "DOC1",
                "title": "T",
                "abstract": "A",
                "doi": "10.1/doi1",
                "year": 2020,
                "journal_full_title": "J",
                "journal": "J",
                "volume": "1",
                "issue": "1",
                "first_page": "1",
                "last_page": "2",
                "pubmed_id": "1",
                "authors": "Auth",
            }
        ]
    }
    with requests_mock.Mocker() as m:
        m.get(f"{base}&document_chembl_id__in=DOC1", json=sample)
        http = HttpClient(timeout=1.0, max_retries=1, rps=0)
        client = ChemblClient(http_client=http)
        df = get_documents(["DOC1"], cfg=cfg, client=client)
    assert df.loc[0, "document_chembl_id"] == "DOC1"
    assert df.loc[0, "pubmed_id"] == "1"


def test_request_json_returns_empty_dict_for_404(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = ApiCfg(timeout_connect=1.0, timeout_read=2.0)
    client = ChemblClient(http_client=HttpClient(timeout=1.0, max_retries=1, rps=0))
    url = "https://www.ebi.ac.uk/chembl/api/data/document.json"

    with requests_mock.Mocker() as mocker:
        mocker.get(url, status_code=404)
        caplog.set_level(logging.WARNING)
        payload = client.request_json(url, cfg=cfg, timeout=cfg.timeout_read)

    assert payload == {}
    assert any("404" in record.message for record in caplog.records)
    last_request = mocker.last_request
    assert last_request is not None
    assert last_request.headers["Accept"] == "application/json"
    assert last_request.headers["User-Agent"] == cfg.user_agent


def test_request_json_raises_for_html_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = ApiCfg()
    client = ChemblClient(http_client=HttpClient(timeout=1.0, max_retries=1, rps=0))
    url = "https://www.ebi.ac.uk/chembl/api/data/document.json"

    with requests_mock.Mocker() as mocker:
        mocker.get(
            url,
            text="<html><body>Error</body></html>",
            headers={"Content-Type": "text/html"},
        )
        caplog.set_level(logging.WARNING)
        with pytest.raises(ValueError):
            client.request_json(url, cfg=cfg, timeout=cfg.timeout_read)

    assert any("Unexpected Content-Type" in record.message for record in caplog.records)


def test_fetch_assay_returns_none_for_404_response() -> None:
    assay_id = "CHEMBL404"
    url = f"https://www.ebi.ac.uk/chembl/api/data/assay/{assay_id}.json"

    with requests_mock.Mocker() as m:
        m.get(url, status_code=404)
        http = HttpClient(timeout=1.0, max_retries=1, rps=0)
        client = ChemblClient(http_client=http)

        result = client.fetch_assay(assay_id)

    assert result is None


def test_fetch_assay_handles_http_error_without_response() -> None:
    class _RaisingClient(HttpClient):
        def __init__(self) -> None:
            super().__init__(timeout=1.0, max_retries=1, rps=0)

        def request(self, method: str, url: str, **kwargs: object) -> object:
            raise requests.HTTPError(
                "404 Client Error: Not Found for url: https://example.invalid/chembl"
            )

    client = ChemblClient(http_client=_RaisingClient())

    assert client.fetch_assay("CHEMBL999") is None


def test_get_documents_batches_requests_and_filters_duplicates() -> None:
    cfg = ApiCfg()

    class _Recorder:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def request_json(
            self, url: str, *, cfg: ApiCfg, timeout: float
        ) -> dict[str, Any]:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            ids_raw = params.get("document_chembl_id__in", [""])[0]
            identifiers = [value for value in ids_raw.split(",") if value]
            limit_raw = params.get("limit", [None])[0]
            self.calls.append(
                {
                    "url": url,
                    "ids": identifiers,
                    "limit": int(limit_raw) if limit_raw is not None else None,
                }
            )
            return {
                "documents": [
                    {
                        "document_chembl_id": doc_id,
                        "title": f"Title {doc_id}",
                        "abstract": f"Abstract {doc_id}",
                        "doi": f"10.{doc_id}/doi{doc_id}",
                        "year": 2024,
                        "journal_full_title": "Journal",
                        "journal": "J",
                        "volume": "1",
                        "issue": "1",
                        "first_page": "1",
                        "last_page": "2",
                        "pubmed_id": doc_id,
                        "authors": "Author",
                    }
                    for doc_id in identifiers
                ]
            }

    recorder = _Recorder()
    client = ChemblClient(http_client=HttpClient(timeout=1.0, max_retries=1, rps=0))
    client.request_json = recorder.request_json  # type: ignore[assignment]

    df = get_documents(
        ["DOC1", "DOC2", "DOC3", "DOC1", "", "#N/A"],
        cfg=cfg,
        client=client,
        chunk_size=2,
    )

    assert [",".join(item["ids"]) for item in recorder.calls] == [
        "DOC1,DOC2",
        "DOC3",
    ]
    assert [item["limit"] for item in recorder.calls] == [2, 1]
    assert df["document_chembl_id"].tolist() == ["DOC1", "DOC2", "DOC3"]


def test_get_documents_returns_all_ids_when_chunk_exceeds_default_limit() -> None:
    cfg = ApiCfg()
    api_default_limit = 20
    requested_ids = [f"DOC{i}" for i in range(api_default_limit + 5)]

    class _LimitedRecorder:
        def __init__(self) -> None:
            self.limits: list[int] = []

        def request_json(
            self, url: str, *, cfg: ApiCfg, timeout: float
        ) -> dict[str, Any]:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            ids_raw = params.get("document_chembl_id__in", [""])[0]
            identifiers = [value for value in ids_raw.split(",") if value]
            limit_raw = params.get("limit", [str(api_default_limit)])[0]
            limit = int(limit_raw)
            self.limits.append(limit)
            subset = identifiers[:limit]
            return {
                "documents": [
                    {
                        "document_chembl_id": doc_id,
                        "title": f"Title {doc_id}",
                        "abstract": f"Abstract {doc_id}",
                        "doi": f"10.{doc_id}/doi{doc_id}",
                        "year": 2024,
                        "journal_full_title": "Journal",
                        "journal": "J",
                        "volume": "1",
                        "issue": "1",
                        "first_page": "1",
                        "last_page": "2",
                        "pubmed_id": doc_id,
                        "authors": "Author",
                    }
                    for doc_id in subset
                ]
            }

    recorder = _LimitedRecorder()
    client = ChemblClient(http_client=HttpClient(timeout=1.0, max_retries=1, rps=0))
    client.request_json = recorder.request_json  # type: ignore[assignment]

    df = get_documents(
        requested_ids,
        cfg=cfg,
        client=client,
        chunk_size=len(requested_ids),
    )

    assert recorder.limits == [len(requested_ids)]
    assert df["document_chembl_id"].tolist() == requested_ids


def test_fetch_many_collects_failed_identifiers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = ChemblClient(http_client=HttpClient(timeout=1.0, max_retries=0, rps=0))
    caplog.set_level(logging.WARNING)

    def failing_fetch(identifier: str) -> dict[str, str]:
        if identifier == "CHEMBL_BAD_REQUEST":
            raise requests.RequestException("boom")
        if identifier == "CHEMBL_BAD_JSON":
            raise ValueError("bad json")
        return {"assay_chembl_id": identifier}

    records, failed = client._fetch_many(  # noqa: SLF001
        ["CHEMBL_OK", "CHEMBL_BAD_REQUEST", "CHEMBL_BAD_JSON"], failing_fetch
    )

    assert records == [{"assay_chembl_id": "CHEMBL_OK"}]
    assert failed == ["CHEMBL_BAD_REQUEST", "CHEMBL_BAD_JSON"]
    messages = "\n".join(record.message for record in caplog.records)
    assert "CHEMBL_BAD_REQUEST" in messages
    assert "CHEMBL_BAD_JSON" in messages


def test_fetch_many_raises_when_failures_exceed_threshold() -> None:
    client = ChemblClient(
        http_client=HttpClient(timeout=1.0, max_retries=0, rps=0),
        failed_ids_error_threshold=1,
    )

    def always_failing(_: str) -> dict[str, str]:
        raise requests.RequestException("network down")

    with pytest.raises(ChemblBatchFetchError) as excinfo:
        client._fetch_many(["CHEMBL1", "CHEMBL2"], always_failing)  # noqa: SLF001

    error = excinfo.value
    assert error.failed_ids == ["CHEMBL1", "CHEMBL2"]
    assert error.threshold == 1


def test_fetch_assay_retries_on_429_without_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = "https://chembl.mock"
    penalties: list[float | None] = []

    def _capture_penalty(self: object, delay: float | None) -> None:
        penalties.append(delay)

    monkeypatch.setattr("tenacity.nap.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        "library.http_client.RateLimiter.apply_penalty", _capture_penalty
    )

    with requests_mock.Mocker() as mocker:
        mocker.get(
            f"{base_url}/assay/CHEMBL1.json",
            [
                {"status_code": 429, "json": {"detail": "limit"}},
                {
                    "status_code": 200,
                    "json": {"assay_chembl_id": "CHEMBL1", "target_chembl_id": "T1"},
                },
            ],
        )
        client = ChemblClient(
            base_url=base_url,
            timeout=1.0,
            max_retries=2,
            rps=0,
            retry_penalty_seconds=2.5,
        )

        payload = client.fetch_assay("CHEMBL1")

    assert payload is not None
    assert payload["assay_chembl_id"] == "CHEMBL1"
    assert mocker.call_count == 2
    assert penalties == [pytest.approx(2.5)]
