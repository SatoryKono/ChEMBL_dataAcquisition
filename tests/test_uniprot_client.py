from typing import Dict
from pathlib import Path
import sys

import requests_mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from library.uniprot_client import NetworkConfig, RateLimitConfig, UniProtClient


def test_fetch_entries_json_batches(requests_mock: requests_mock.Mocker) -> None:
    client = UniProtClient(
        base_url="https://rest.uniprot.org/uniprotkb",
        fields="",
        network=NetworkConfig(timeout_sec=1, max_retries=1),
        rate_limit=RateLimitConfig(rps=1000),
    )
    url = "https://rest.uniprot.org/uniprotkb/stream"
    requests_mock.get(
        url,
        json={
            "results": [
                {"primaryAccession": "P1"},
                {"primaryAccession": "P2"},
            ]
        },
    )
    res: Dict[str, Dict[str, str]] = client.fetch_entries_json(["P1", "P2"])
    assert requests_mock.call_count == 1
    assert set(res.keys()) == {"P1", "P2"}
