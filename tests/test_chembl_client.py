"""Tests for the ChEMBL document client."""

from __future__ import annotations

import sys
from pathlib import Path

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
