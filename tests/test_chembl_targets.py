from pathlib import Path
import sys
import json
import logging
from typing import Any

import pytest
import requests_mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import library.chembl_targets as chembl_targets
from library.chembl_targets import TargetConfig, fetch_targets


def test_fetch_targets_batches(requests_mock: requests_mock.Mocker) -> None:
    cfg = TargetConfig(base_url="https://example.org")
    url = "https://example.org/target"

    def _first_chunk_matcher(request: Any) -> bool:
        url = request.url
        return "CHEMBL1%2CCHEMBL2" in url or "CHEMBL1,CHEMBL2" in url

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
        additional_matcher=_first_chunk_matcher,
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


def test_fetch_targets_error_reporting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class DummyResponse:
        def __init__(self, url: str) -> None:
            self.status_code = 500
            self.url = url

        def json(self) -> dict[str, str]:
            return {}

    class DummyClient:
        def request(
            self, method: str, url: str, params: dict[str, str]
        ) -> DummyResponse:
            return DummyResponse(url)

    monkeypatch.setattr(chembl_targets, "HttpClient", lambda **_: DummyClient())

    cfg = TargetConfig(base_url="https://example.org")
    output_path = tmp_path / "targets.csv"
    caplog.set_level(logging.WARNING, logger=chembl_targets.__name__)

    with pytest.raises(RuntimeError) as excinfo:
        fetch_targets(
            [
                "CHEMBL1",
            ],
            cfg,
            batch_size=1,
            output_path=output_path,
        )

    assert "Failed to fetch 1 chunk" in str(excinfo.value)

    error_path = output_path.with_suffix(f"{output_path.suffix}.errors.json")
    assert error_path.exists()
    payload = json.loads(error_path.read_text(encoding="utf-8"))
    assert len(payload) == 1
    entry = payload[0]
    assert entry["chunk_index"] == 0
    assert entry["identifiers"] == ["CHEMBL1"]
    assert entry["status_code"] == 500
    assert any(
        "Chunk 0" in record.getMessage() and "500" in record.getMessage()
        for record in caplog.records
    )
