from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest
import requests

from chembl2uniprot import map_chembl_to_uniprot
from chembl2uniprot.mapping import RateLimiter, _poll_job

DATA_DIR = Path(__file__).parent / "data"
CONFIG_DIR = DATA_DIR / "config"
CSV_DIR = DATA_DIR / "csv"

CONFIG = CONFIG_DIR / "valid.yaml"
SCHEMA = CONFIG_DIR / "config.schema.json"
INPUT = CSV_DIR / "input.csv"
INPUT_SINGLE = CSV_DIR / "input_single.csv"


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    schema = tmp_path / "config.schema.json"
    cfg.write_text(CONFIG.read_text())
    schema.write_text(SCHEMA.read_text())
    return cfg


def read_output(path: Path) -> list[str | None]:
    df = pd.read_csv(path)
    out: list[str | None] = []
    for value in df["mapped_uniprot_id"]:
        out.append(None if pd.isna(value) else value)
    return out


def test_success_mapping_single_batch(
    requests_mock, tmp_path: Path, config_path: Path
) -> None:
    run_url = "https://rest.uniprot.org/idmapping/run"
    status_url = "https://rest.uniprot.org/idmapping/status/123"
    results_url = "https://rest.uniprot.org/idmapping/results/123"
    requests_mock.post(run_url, json={"jobId": "123"})
    requests_mock.get(status_url, json={"jobStatus": "FINISHED"})
    requests_mock.get(
        results_url,
        json={
            "results": [
                {"from": "CHEMBL1", "to": "P1"},
                {"from": "CHEMBL2", "to": "P2"},
            ]
        },
    )

    out = tmp_path / "out.csv"
    map_chembl_to_uniprot(INPUT, out, config_path)
    assert read_output(out) == ["P1", "P2"]


def test_no_mapping(requests_mock, tmp_path: Path, config_path: Path) -> None:
    run_url = "https://rest.uniprot.org/idmapping/run"
    status_url = "https://rest.uniprot.org/idmapping/status/1"
    results_url = "https://rest.uniprot.org/idmapping/results/1"
    requests_mock.post(run_url, json={"jobId": "1"})
    requests_mock.get(status_url, json={"jobStatus": "FINISHED"})
    requests_mock.get(results_url, json={"results": []})
    out = tmp_path / "out.csv"
    map_chembl_to_uniprot(INPUT, out, config_path)
    assert read_output(out) == [None, None]


def test_multiple_mappings(requests_mock, tmp_path: Path, config_path: Path) -> None:
    run_url = "https://rest.uniprot.org/idmapping/run"
    status_url = "https://rest.uniprot.org/idmapping/status/1"
    results_url = "https://rest.uniprot.org/idmapping/results/1"
    requests_mock.post(run_url, json={"jobId": "1"})
    requests_mock.get(status_url, json={"jobStatus": "FINISHED"})
    requests_mock.get(
        results_url,
        json={
            "results": [
                {"from": "CHEMBL1", "to": "P1"},
                {"from": "CHEMBL1", "to": "P2"},
            ]
        },
    )
    out = tmp_path / "out.csv"
    map_chembl_to_uniprot(INPUT_SINGLE, out, config_path)
    assert read_output(out) == ["P1|P2"]


def test_retry_on_server_error(
    requests_mock, tmp_path: Path, config_path: Path
) -> None:
    run_url = "https://rest.uniprot.org/idmapping/run"
    status_url = "https://rest.uniprot.org/idmapping/status/1"
    results_url = "https://rest.uniprot.org/idmapping/results/1"
    requests_mock.post(run_url, [{"status_code": 500}, {"json": {"jobId": "1"}}])
    requests_mock.get(status_url, json={"jobStatus": "FINISHED"})
    requests_mock.get(results_url, json={"results": []})
    out = tmp_path / "out.csv"
    map_chembl_to_uniprot(INPUT, out, config_path)
    assert read_output(out) == [None, None]


def test_config_validation_error(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    schema = tmp_path / "config.schema.json"
    schema.write_text(SCHEMA.read_text())
    cfg.write_text("io: {}")  # invalid
    with pytest.raises(ValueError):
        map_chembl_to_uniprot(INPUT, tmp_path / "out.csv", cfg)


def test_poll_job_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SimpleNamespace(
        base_url="https://rest.uniprot.org",
        id_mapping=SimpleNamespace(status_endpoint="/idmapping/status"),
        polling=SimpleNamespace(interval_sec=0),
    )

    def raise_timeout(*_: object, **__: object) -> None:
        raise requests.Timeout

    monkeypatch.setattr("chembl2uniprot.mapping._request_with_retry", raise_timeout)

    with pytest.raises(requests.Timeout):
        _poll_job(
            "1",
            cfg,
            RateLimiter(0),
            timeout=0.1,
            retry_cfg=SimpleNamespace(max_attempts=1, backoff_sec=0),
        )


def test_deterministic_csv(requests_mock, tmp_path: Path, config_path: Path) -> None:
    run_url = "https://rest.uniprot.org/idmapping/run"
    status_url = "https://rest.uniprot.org/idmapping/status/1"
    results_url = "https://rest.uniprot.org/idmapping/results/1"

    requests_mock.post(run_url, json={"jobId": "1"})
    requests_mock.get(status_url, json={"jobStatus": "FINISHED"})
    requests_mock.get(
        results_url,
        [
            {"json": {"results": [{"from": "CHEMBL1", "to": "P1"}]}},
            {"json": {"results": [{"to": "P1", "from": "CHEMBL1"}]}},
        ],
    )

    out1 = tmp_path / "out1.csv"
    out2 = tmp_path / "out2.csv"
    map_chembl_to_uniprot(INPUT_SINGLE, out1, config_path)
    map_chembl_to_uniprot(INPUT_SINGLE, out2, config_path)

    assert out1.read_bytes() == out2.read_bytes()
