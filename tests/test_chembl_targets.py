import logging
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlsplit

import pytest
import requests
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
        additional_matcher=lambda r: parse_qs(urlsplit(r.url).query).get(
            "target_chembl_id__in"
        )
        == ["CHEMBL1,CHEMBL2"],
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
        additional_matcher=lambda r: parse_qs(urlsplit(r.url).query).get(
            "target_chembl_id__in"
        )
        == ["CHEMBL3"],
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


def test_fetch_targets_raises_on_server_error(
    requests_mock: requests_mock.Mocker,
) -> None:
    cfg = TargetConfig(base_url="https://example.org", max_retries=1)
    url = "https://example.org/target"
    requests_mock.get(url, status_code=500, text="server error")
    with pytest.raises(requests.HTTPError):
        fetch_targets(["CHEMBL1"], cfg, batch_size=1)


def test_fetch_targets_raises_on_not_found(
    requests_mock: requests_mock.Mocker,
) -> None:
    cfg = TargetConfig(base_url="https://example.org")
    url = "https://example.org/target"
    requests_mock.get(url, status_code=404, json={"detail": "missing"})
    with pytest.raises(requests.HTTPError):
        fetch_targets(["CHEMBL1"], cfg, batch_size=1)


def test_fetch_targets_warns_on_empty_response(
    requests_mock: requests_mock.Mocker, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = TargetConfig(base_url="https://example.org")
    url = "https://example.org/target"
    requests_mock.get(url, status_code=200, json={"targets": []})
    with caplog.at_level(logging.WARNING):
        df = fetch_targets(["CHEMBL1"], cfg, batch_size=1)
    assert "Empty target payload" in caplog.text
    assert df.shape[0] == 1
