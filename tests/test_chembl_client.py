"""Tests for the ChEMBL document client."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import requests  # type: ignore[import-untyped]
import requests_mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from library.http_client import HttpClient  # type: ignore  # noqa: E402
from library.chembl_client import ApiCfg, ChemblClient, get_documents  # type: ignore  # noqa: E402


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
            self.calls: list[str] = []

        def request_json(self, url: str, *, cfg: ApiCfg, timeout: float) -> dict[str, Any]:
            self.calls.append(url)
            ids_part = url.split("document_chembl_id__in=")[1]
            identifiers = ids_part.split(",")
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

    assert [call.split("document_chembl_id__in=")[1] for call in recorder.calls] == [
        "DOC1,DOC2",
        "DOC3",
    ]
    assert df["document_chembl_id"].tolist() == ["DOC1", "DOC2", "DOC3"]
