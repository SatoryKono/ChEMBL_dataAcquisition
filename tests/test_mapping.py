from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from chembl2uniprot import map_chembl_to_uniprot

DATA_DIR = Path(__file__).parent / "data"
CONFIG = DATA_DIR / "config.yaml"
SCHEMA = DATA_DIR / "config.schema.json"
INPUT = DATA_DIR / "input.csv"


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
    map_chembl_to_uniprot(DATA_DIR / "input_single.csv", out, config_path)
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


def test_retry_on_rate_limit(
    requests_mock, tmp_path: Path, config_path: Path, monkeypatch
) -> None:
    run_url = "https://rest.uniprot.org/idmapping/run"
    status_url = "https://rest.uniprot.org/idmapping/status/1"
    results_url = "https://rest.uniprot.org/idmapping/results/1"
    requests_mock.post(
        run_url,
        [
            {"status_code": 429, "headers": {"Retry-After": "1"}},
            {"json": {"jobId": "1"}},
        ],
    )
    requests_mock.get(status_url, json={"jobStatus": "FINISHED"})
    requests_mock.get(results_url, json={"results": []})

    monkeypatch.setattr("chembl2uniprot.mapping.RateLimiter.wait", lambda self: None)
    calls: list[float] = []

    def fake_sleep(seconds: float) -> None:
        calls.append(seconds)

    monkeypatch.setattr("chembl2uniprot.mapping.time.sleep", fake_sleep)

    out = tmp_path / "out.csv"
    map_chembl_to_uniprot(INPUT, out, config_path)
    assert read_output(out) == [None, None]
    assert 1 in calls


def test_config_validation_error(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    schema = tmp_path / "config.schema.json"
    schema.write_text(SCHEMA.read_text())
    cfg.write_text("io: {}")  # invalid
    with pytest.raises(ValueError):
        map_chembl_to_uniprot(INPUT, tmp_path / "out.csv", cfg)
