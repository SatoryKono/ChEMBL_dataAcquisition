from __future__ import annotations

import requests_mock


from library.gtop_client import GtoPClient, GtoPConfig, resolve_target
from library.gtop_normalize import normalise_interactions, normalise_synonyms



def _client() -> GtoPClient:
    cfg = GtoPConfig(base_url="http://test", timeout_sec=0, max_retries=1, rps=0)
    return GtoPClient(cfg)


def test_resolve_uniprot_id(requests_mock: requests_mock.Mocker) -> None:
    client = _client()
    url = "http://test/targets"
    requests_mock.get(url, json=[{"targetId": 1, "name": "t", "species": "Human"}])
    target = resolve_target(client, "P35498", "uniprot_id")
    assert target and target["targetId"] == 1


def test_resolve_gene_symbol_prefers_human(requests_mock: requests_mock.Mocker) -> None:
    client = _client()
    url = "http://test/targets"
    payload = [
        {"targetId": 1, "name": "mouse", "species": "Mouse"},
        {"targetId": 2, "name": "human", "species": "Human"},
    ]
    requests_mock.get(url, json=payload)
    target = resolve_target(client, "ADRA1A", "gene_symbol")
    assert target and target["targetId"] == 2


def test_empty_block_returns_empty_df(requests_mock: requests_mock.Mocker) -> None:
    client = _client()
    url = "http://test/targets/1/synonyms"
    requests_mock.get(url, status_code=204)
    data = client.fetch_target_endpoint(1, "synonyms")
    df = normalise_synonyms(1, data)
    assert df.empty


def test_interactions_filter_params(requests_mock: requests_mock.Mocker) -> None:
    client = _client()
    url = "http://test/targets/1/interactions"
    requests_mock.get(
        url, json=[{"ligandId": 10, "affinity": 8, "affinityParameter": "pKi"}]
    )
    data = client.fetch_target_endpoint(
        1,
        "interactions",
        params={"affinityType": "pKi", "affinity": 7, "approved": True},
    )
    df = normalise_interactions(1, data)
    assert list(df["ligandId"]) == [10]
    assert requests_mock.last_request.qs == {
        "affinitytype": ["pki"],
        "affinity": ["7"],
        "approved": ["true"],
    }
