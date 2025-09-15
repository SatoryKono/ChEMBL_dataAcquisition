from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from library.orthologs import EnsemblHomologyClient
from library.uniprot_normalize import extract_ensembl_gene_ids
from library.uniprot_client import NetworkConfig, RateLimitConfig


class DummyResponse:
    def __init__(self, data: Any) -> None:
        self._data = data
        self.status_code = 200

    def json(self) -> Any:  # pragma: no cover - simple
        return self._data


def test_extract_ensembl_gene_ids() -> None:
    entry = json.loads(Path("tests/data/uniprot_with_ensembl.json").read_text())
    ids = extract_ensembl_gene_ids(entry)
    assert ids == ["ENSG00000144285"]


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
