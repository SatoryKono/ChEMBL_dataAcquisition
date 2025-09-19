import json
import threading
import time
from pathlib import Path

import pandas as pd
import pytest
import requests_mock

from library.http_client import CacheConfig, HttpClient  # type: ignore  # noqa: E402
from library.uniprot_enrich.enrich import UniProtClient  # type: ignore  # noqa: E402
from uniprot_enrich import enrich_uniprot

MOCK_ENTRY = {
    "primaryAccession": "P12345",
    "uniProtkbId": "TEST_HUMAN",
    "entryType": "Reviewed",
    "secondaryAccessions": ["Q99999"],
    "proteinDescription": {
        "recommendedName": {
            "fullName": {"value": "Test protein"},
            "ecNumbers": [{"value": "1.2.3.4"}],
        },
        "alternativeNames": [
            {"fullName": {"value": "Alt1"}},
            {"fullName": {"value": "Alt2"}},
        ],
    },
    "genes": [{"geneName": {"value": "TP"}}],
    "organism": {
        "taxonId": 9606,
        "lineage": ["Eukaryota", "Metazoa", "Chordata", "Testus"],
    },
    "comments": [
        {
            "commentType": "SUBCELLULAR_LOCATION",
            "subcellularLocations": [
                {
                    "location": {"value": "Cytoplasm"},
                    "topology": {"value": "Peripheral membrane protein"},
                }
            ],
        },
        {
            "commentType": "ALTERNATIVE_PRODUCTS",
            "isoforms": [
                {
                    "name": {"value": "Isoform 1"},
                    "id": "P12345-1",
                    "synonyms": [{"value": "Iso1"}],
                }
            ],
        },
        {
            "commentType": "CATALYTIC_ACTIVITY",
            "reaction": {"name": "A + B = C", "ecNumber": "1.2.3.4"},
        },
    ],
    "features": [
        {
            "type": "Glycosylation",
            "description": "N-linked (GlcNAc) asparagine",
            "location": {"start": {"value": 2}, "end": {"value": 2}},
        },
        {
            "type": "Topological domain",
            "description": "Extracellular",
            "location": {"start": {"value": 1}, "end": {"value": 3}},
        },
        {
            "type": "Transmembrane region",
            "location": {"start": {"value": 10}, "end": {"value": 30}},
        },
        {
            "type": "Intramembrane region",
            "location": {"start": {"value": 40}, "end": {"value": 50}},
        },
        {
            "type": "Modified residue",
            "description": "Phosphothreonine",
            "location": {"start": {"value": 5}, "end": {"value": 5}},
        },
    ],
    "uniProtKBCrossReferences": [
        {
            "database": "HGNC",
            "id": "HGNC:1",
            "properties": [{"key": "Name", "value": "TP"}],
        },
        {"database": "GuidetoPHARMACOLOGY", "id": "G123"},
        {
            "database": "GO",
            "id": "GO:0003677",
            "properties": [{"key": "GoTerm", "value": "F:DNA binding"}],
        },
        {
            "database": "GO",
            "id": "GO:0005634",
            "properties": [{"key": "GoTerm", "value": "C:nucleus"}],
        },
        {"database": "PROSITE", "id": "PS12345"},
        {"database": "Pfam", "id": "PF00001"},
    ],
}

MOCK_SECONDARY = {
    "primaryAccession": "Q99999",
    "proteinDescription": {
        "recommendedName": {"fullName": {"value": "Secondary protein"}}
    },
}


@pytest.fixture()
def data_file(tmp_path: Path) -> Path:
    src = Path("tests/data/uniprot_sample.csv")
    dst = tmp_path / "uniprot_sample.csv"
    dst.write_text(src.read_text())
    return dst


def test_enrich_uniprot(data_file: Path) -> None:
    with requests_mock.Mocker() as m:
        m.get(
            "https://rest.uniprot.org/uniprotkb/P12345?format=json",
            text=json.dumps(MOCK_ENTRY),
        )
        m.get(
            "https://rest.uniprot.org/uniprotkb/Q99999?format=json",
            text=json.dumps(MOCK_SECONDARY),
        )
        m.get(
            "https://rest.uniprot.org/uniprotkb/P99999?format=json",
            status_code=404,
        )
        enrich_uniprot(str(data_file))
        assert m.call_count == 3
    df = pd.read_csv(data_file, dtype=str).fillna("")
    assert df.loc[0, "recommended_name"] == "Test protein"
    assert df.loc[0, "synonyms"] == "Alt1|Alt2"
    assert df.loc[1, "recommended_name"] == ""
    assert df.loc[2, "gene_name"] == "TP"
    assert df.loc[0, "secondary_accession_names"] == "Secondary protein"
    assert df.loc[0, "PROSITE"] == "PS12345"
    assert df.loc[0, "reaction_ec_numbers"] == "1.2.3.4"
    assert df.loc[0, "subcellular_location"] == "Cytoplasm"
    assert "Peripheral membrane protein" in df.loc[0, "topology"]
    assert df.loc[0, "transmembrane"] == "1"
    assert df.loc[0, "intramembrane"] == "1"
    # column order
    expected_cols = ["uniprot_id", "other"] + list(
        enrich_uniprot.__globals__["OUTPUT_COLUMNS"]
    )
    assert list(df.columns) == expected_cols


def test_fetch_all_uses_http_cache(tmp_path: Path) -> None:
    cache = CacheConfig(
        enabled=True,
        path=str(tmp_path / "cache" / "uniprot"),
        ttl_seconds=60,
    )
    http_client = HttpClient(timeout=1.0, max_retries=1, rps=0.0, cache_config=cache)
    client = UniProtClient(http_client=http_client)
    url = "https://rest.uniprot.org/uniprotkb/P12345?format=json"
    with requests_mock.Mocker() as m:
        m.get(url, json=MOCK_ENTRY)
        m.get(
            "https://rest.uniprot.org/uniprotkb/Q99999?format=json",
            json=MOCK_SECONDARY,
        )
        client.fetch_all(["P12345"])
        assert m.call_count == 2
        # Clear in-memory cache to validate HTTP layer caching behaviour.
        client.cache.clear()
        client.fetch_all(["P12345"])
        assert m.call_count == 2


def test_fetch_all_uses_thread_local_http_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread_to_client: dict[int, int] = {}
    client_to_thread: dict[int, int] = {}

    class DummyResponse:
        def __init__(self, accession: str) -> None:
            self.status_code = 200
            self._accession = accession
            self.headers: dict[str, str] = {}

        def json(self) -> dict[str, str]:
            return {"primaryAccession": self._accession}

    def fake_request(self: HttpClient, method: str, url: str, **_: object) -> DummyResponse:
        thread_id = threading.get_ident()
        client_id = id(self)
        bound_client = thread_to_client.setdefault(thread_id, client_id)
        assert bound_client == client_id
        bound_thread = client_to_thread.setdefault(client_id, thread_id)
        assert bound_thread == thread_id
        assert self.timeout == (0.5, 0.5)
        assert self.max_retries == 2
        assert self.rate_limiter.rps == 3.0
        time.sleep(0.01)
        accession = url.rsplit("/", 1)[-1].split("?")[0]
        return DummyResponse(accession)

    monkeypatch.setattr(
        "library.uniprot_enrich.enrich.HttpClient.request", fake_request
    )

    client = UniProtClient(
        max_workers=2,
        request_timeout=0.5,
        max_retries=2,
        rate_limit_rps=3.0,
    )
    accessions = ["A0A000", "B0B000", "C0C000", "D0D000"]
    results = client.fetch_all(accessions)

    assert set(results) == set(accessions)
    assert len(thread_to_client) >= 2
    assert len(thread_to_client) == len(set(thread_to_client.values()))
    assert len(client_to_thread) == len(thread_to_client)
