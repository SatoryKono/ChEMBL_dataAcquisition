from pathlib import Path
import sys

import requests_mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from library.chembl_targets import TargetConfig, fetch_targets


def test_fetch_targets_batches(requests_mock: requests_mock.Mocker) -> None:
    cfg = TargetConfig(base_url="https://example.org")
    url = "https://example.org/target"
    requests_mock.get(
        url,
        json={
            "targets": [
                {
                    "target_chembl_id": "CHEMBL1",
                    "target_components": [],
                    "cross_references": [],
                },
                {
                    "target_chembl_id": "CHEMBL2",
                    "target_components": [],
                    "cross_references": [],
                },
            ]
        },
        additional_matcher=lambda r: "CHEMBL1,CHEMBL2" in r.url,
    )
    requests_mock.get(
        url,
        json={
            "targets": [
                {
                    "target_chembl_id": "CHEMBL3",
                    "target_components": [],
                    "cross_references": [],
                }
            ]
        },
        additional_matcher=lambda r: "CHEMBL3" in r.url and "CHEMBL1" not in r.url,
    )
    df = fetch_targets(
        [
            "CHEMBL1",
            "CHEMBL2",
            "CHEMBL3",
        ],
        cfg,
        batch_size=2,
    )
    assert requests_mock.call_count == 2
    assert set(df["target_chembl_id"]) == {"CHEMBL1", "CHEMBL2", "CHEMBL3"}
