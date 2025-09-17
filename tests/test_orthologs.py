from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from library.orthologs import EnsemblHomologyClient  # type: ignore  # noqa: E402
from library.uniprot_normalize import extract_ensembl_gene_ids  # type: ignore  # noqa: E402
from library.uniprot_client import NetworkConfig, RateLimitConfig  # type: ignore  # noqa: E402


class DummyResponse:
    def __init__(self, data: Any) -> None:
        self._data = data
        self.status_code = 200

    def json(self) -> Any:  # pragma: no cover - simple
        return self._data


class DummyHttpClient:
    def __init__(self, responses: list[requests.Response | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        self.calls.append((method, url, kwargs))
        if not self.responses:
            raise AssertionError("No responses configured")
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def make_response(
    status: int,
    payload: Any | None = None,
    *,
    headers: dict[str, str] | None = None,
) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.url = "https://example.org/test"
    if headers:
        response.headers.update(headers)
    if payload is not None:
        response._content = json.dumps(payload).encode("utf-8")
        response.headers.setdefault("Content-Type", "application/json")
    else:
        response._content = b""
    return response


def test_extract_ensembl_gene_ids() -> None:
    entry = json.loads(Path("tests/data/uniprot_with_ensembl.json").read_text())
    ids = extract_ensembl_gene_ids(entry)
    assert ids == ["ENSG00000144285"]


def test_ensembl_request_success() -> None:
    response = make_response(200, {"hello": "world"})
    http_client = DummyHttpClient([response])
    client = EnsemblHomologyClient(
        base_url="https://example.org",
        network=NetworkConfig(timeout_sec=1, max_retries=1),
        rate_limit=RateLimitConfig(rps=1),
        http_client=http_client,
    )
    result = client._request("https://example.org/homology", [("a", "b")])
    assert result is response
    assert len(http_client.calls) == 1
    method, url, kwargs = http_client.calls[0]
    assert method == "get"
    assert url == "https://example.org/homology"
    assert kwargs["headers"]["Accept"] == "application/json"
    assert kwargs["params"] == [("a", "b")]


def test_ensembl_request_handles_server_error() -> None:
    response = make_response(500)
    error = requests.HTTPError(response=response)
    http_client = DummyHttpClient([error])
    client = EnsemblHomologyClient(
        base_url="https://example.org",
        network=NetworkConfig(timeout_sec=1, max_retries=1),
        rate_limit=RateLimitConfig(rps=1),
        http_client=http_client,
    )
    result = client._request("https://example.org/homology", [])
    assert result is None
    assert len(http_client.calls) == 1


def test_ensembl_request_handles_retry_after_header() -> None:
    response = make_response(429, headers={"Retry-After": "1"})
    error = requests.HTTPError(response=response)
    http_client = DummyHttpClient([error])
    client = EnsemblHomologyClient(
        base_url="https://example.org",
        network=NetworkConfig(timeout_sec=1, max_retries=1),
        rate_limit=RateLimitConfig(rps=1),
        http_client=http_client,
    )
    result = client._request("https://example.org/homology", [])
    assert result is None


def test_ensembl_homology_client_parsing(monkeypatch: Any) -> None:
    data = json.loads(Path("tests/data/ensembl_homology.json").read_text())
    client = EnsemblHomologyClient(
        base_url="https://example.org",
        network=NetworkConfig(timeout_sec=1, max_retries=1),
        rate_limit=RateLimitConfig(rps=100),
    )

    def fake_request(url: str, params: Any) -> DummyResponse:
        return DummyResponse(data)

    monkeypatch.setattr(client, "_request", fake_request)
    monkeypatch.setattr(
        client,
        "_lookup_gene_symbol",
        lambda gid: {"ENSMUSG00000002413": "Braf", "ENSRNOG00000010957": "Braf"}[gid],
    )
    monkeypatch.setattr(
        client,
        "_lookup_uniprot",
        lambda gid: {
            "ENSMUSG00000002413": "P12345",
            "ENSRNOG00000010957": "Q54321",
        }.get(gid, ""),
    )

    orthologs = client.get_orthologs("ENSG00000144285", ["mouse", "rat"])
    assert len(orthologs) == 2
    assert orthologs[0].target_species == "mus_musculus"
    assert orthologs[0].target_gene_symbol == "Braf"
    assert orthologs[0].perc_id == 84.3
    assert orthologs[0].dn == 0.1


def test_ensembl_homology_client_handles_empty_payload(monkeypatch: Any) -> None:
    client = EnsemblHomologyClient(
        base_url="https://example.org",
        network=NetworkConfig(timeout_sec=1, max_retries=1),
        rate_limit=RateLimitConfig(rps=100),
    )

    def fake_request(url: str, params: Any) -> DummyResponse:
        return DummyResponse({"data": []})

    monkeypatch.setattr(client, "_request", fake_request)

    orthologs = client.get_orthologs("ENSG00000144285", ["mouse", "rat"])
    assert orthologs == []
