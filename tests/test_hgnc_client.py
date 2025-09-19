from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hgnc_client import HGNCClient, load_config, map_uniprot_to_hgnc

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"


class FakeClock:
    """Deterministic clock used to capture retry delays."""

    def __init__(self) -> None:
        self.monotonic_time = 0.0
        self.wall_time = 1_000_000.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.monotonic_time += seconds
        self.wall_time += seconds

    def monotonic(self) -> float:
        return self.monotonic_time

    def time(self) -> float:
        return self.wall_time


def _patch_clock(monkeypatch, clock: FakeClock) -> None:
    """Patch ``time``-related functions for deterministic retry timing."""

    monkeypatch.setattr("tenacity.nap.time.sleep", clock.sleep)
    monkeypatch.setattr("library.http_client.time.sleep", clock.sleep)
    monkeypatch.setattr("library.http_client.time.monotonic", clock.monotonic)
    monkeypatch.setattr("library.http_client.time.time", clock.time)


def _write_csv(path: Path, values: list[str]) -> None:
    df = pd.DataFrame({"uniprot_id": values})
    df.to_csv(path, index=False)


def _read_output(path: Path) -> list[dict[str, str]]:
    df = pd.read_csv(path).fillna("")
    return df.to_dict(orient="records")


def test_valid_uniprot_id(requests_mock, tmp_path: Path) -> None:
    in_csv = tmp_path / "in.csv"
    _write_csv(in_csv, ["P35348"])
    out_csv = tmp_path / "out.csv"

    hgnc_url = "https://rest.genenames.org/fetch/uniprot_ids/P35348"
    requests_mock.get(
        hgnc_url,
        json={
            "response": {
                "docs": [
                    {
                        "symbol": "ADRA1A",
                        "name": "adrenoceptor alpha 1A",
                        "hgnc_id": "HGNC:277",
                    }
                ]
            }
        },
    )
    uniprot_url = "https://rest.uniprot.org/uniprotkb/P35348.json"
    requests_mock.get(
        uniprot_url,
        json={
            "proteinDescription": {
                "recommendedName": {
                    "fullName": {"value": "Alpha-1A adrenergic receptor"}
                }
            }
        },
    )

    map_uniprot_to_hgnc(in_csv, out_csv, CONFIG, config_section="hgnc")
    assert _read_output(out_csv) == [
        {
            "uniprot_id": "P35348",
            "hgnc_id": "HGNC:277",
            "gene_symbol": "ADRA1A",
            "gene_name": "adrenoceptor alpha 1A",
            "protein_name": "Alpha-1A adrenergic receptor",
        }
    ]


def test_non_human_uniprot_id(requests_mock, tmp_path: Path) -> None:
    in_csv = tmp_path / "in.csv"
    _write_csv(in_csv, ["Q91X72"])
    out_csv = tmp_path / "out.csv"

    hgnc_url = "https://rest.genenames.org/fetch/uniprot_ids/Q91X72"
    requests_mock.get(hgnc_url, json={"response": {"docs": []}})
    uniprot_url = "https://rest.uniprot.org/uniprotkb/Q91X72.json"
    uniprot_mock = requests_mock.get(uniprot_url, json={})

    map_uniprot_to_hgnc(in_csv, out_csv, CONFIG, config_section="hgnc")
    assert _read_output(out_csv) == [
        {
            "uniprot_id": "Q91X72",
            "hgnc_id": "",
            "gene_symbol": "",
            "gene_name": "",
            "protein_name": "",
        }
    ]
    assert uniprot_mock.call_count == 0


def test_duplicate_input_ids(requests_mock, tmp_path: Path) -> None:
    in_csv = tmp_path / "in.csv"
    _write_csv(in_csv, ["P35348", "P35348"])
    out_csv = tmp_path / "out.csv"

    hgnc_url = "https://rest.genenames.org/fetch/uniprot_ids/P35348"
    hgnc_mock = requests_mock.get(
        hgnc_url,
        json={
            "response": {
                "docs": [
                    {
                        "symbol": "ADRA1A",
                        "name": "adrenoceptor alpha 1A",
                        "hgnc_id": "HGNC:277",
                    }
                ]
            }
        },
    )
    uniprot_url = "https://rest.uniprot.org/uniprotkb/P35348.json"
    uniprot_mock = requests_mock.get(
        uniprot_url,
        json={
            "proteinDescription": {
                "recommendedName": {
                    "fullName": {"value": "Alpha-1A adrenergic receptor"}
                }
            }
        },
    )

    map_uniprot_to_hgnc(in_csv, out_csv, CONFIG, config_section="hgnc")
    assert _read_output(out_csv) == [
        {
            "uniprot_id": "P35348",
            "hgnc_id": "HGNC:277",
            "gene_symbol": "ADRA1A",
            "gene_name": "adrenoceptor alpha 1A",
            "protein_name": "Alpha-1A adrenergic receptor",
        },
        {
            "uniprot_id": "P35348",
            "hgnc_id": "HGNC:277",
            "gene_symbol": "ADRA1A",
            "gene_name": "adrenoceptor alpha 1A",
            "protein_name": "Alpha-1A adrenergic receptor",
        },
    ]
    assert hgnc_mock.call_count == 1
    assert uniprot_mock.call_count == 1


def test_parallel_requests_call_count(requests_mock, tmp_path: Path) -> None:
    in_csv = tmp_path / "in.csv"
    _write_csv(in_csv, ["P35348", "P24530"])
    out_csv = tmp_path / "out.csv"

    for accession, gene_symbol in [
        ("P35348", "ADRA1A"),
        ("P24530", "ADRA1D"),
    ]:
        hgnc_url = f"https://rest.genenames.org/fetch/uniprot_ids/{accession}"
        requests_mock.get(
            hgnc_url,
            json={
                "response": {
                    "docs": [
                        {
                            "symbol": gene_symbol,
                            "name": "placeholder",
                            "hgnc_id": "HGNC:0000",
                        }
                    ]
                }
            },
        )
        uniprot_url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
        requests_mock.get(
            uniprot_url,
            json={
                "proteinDescription": {
                    "recommendedName": {"fullName": {"value": "protein"}}
                }
            },
        )

    map_uniprot_to_hgnc(in_csv, out_csv, CONFIG, config_section="hgnc")
    expected_requests = len({"P35348", "P24530"}) * 2
    assert requests_mock.call_count == expected_requests


def test_hgnc_client_recovers_after_retry_after(
    monkeypatch, requests_mock, caplog
) -> None:
    clock = FakeClock()
    _patch_clock(monkeypatch, clock)
    caplog.set_level("WARNING", "library.http_client")

    cfg = load_config(CONFIG, section="hgnc")
    with HGNCClient(cfg) as client:
        accession = "P35348"
        hgnc_url = f"https://rest.genenames.org/fetch/uniprot_ids/{accession}"
        requests_mock.get(
            hgnc_url,
            [
                {
                    "status_code": 429,
                    "headers": {"Retry-After": "1.5"},
                    "json": {"response": {"docs": []}},
                },
                {
                    "status_code": 200,
                    "json": {
                        "response": {
                            "docs": [
                                {
                                    "symbol": "ADRA1A",
                                    "name": "adrenoceptor alpha 1A",
                                    "hgnc_id": "HGNC:277",
                                }
                            ]
                        }
                    },
                },
            ],
        )
        uniprot_url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
        requests_mock.get(
            uniprot_url,
            json={
                "proteinDescription": {
                    "recommendedName": {
                        "fullName": {"value": "Alpha-1A adrenergic receptor"}
                    }
                }
            },
        )

        record = client.fetch(accession)

    assert record.hgnc_id == "HGNC:277"
    assert record.gene_symbol == "ADRA1A"
    hgnc_calls = [
        call for call in requests_mock.request_history if call.url == hgnc_url
    ]
    assert len(hgnc_calls) == 2
    assert any(pytest.approx(delay, rel=1e-3) == 1.5 for delay in clock.sleeps)
    assert "retrying after 1.50 seconds" in caplog.text
