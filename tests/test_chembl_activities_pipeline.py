from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import pytest
import requests_mock as requests_mock_lib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

normalize_activities = importlib.import_module(
    "library.normalize_activities"
).normalize_activities
validate_activities = importlib.import_module(
    "library.activity_validation"
).validate_activities
ChemblClient = importlib.import_module("library.chembl_client").ChemblClient
get_activities = importlib.import_module("library.chembl_library").get_activities
read_ids = importlib.import_module("library.io").read_ids
CsvConfig = importlib.import_module("library.io_utils").CsvConfig
chembl_activities_main = importlib.import_module("scripts.chembl_activities_main").main


def test_read_ids_limit(tmp_path: Path) -> None:
    source = Path(__file__).parent / "data" / "activities_input.csv"
    target = tmp_path / "ids.csv"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    cfg = CsvConfig(sep=",", encoding="utf-8")
    ids = list(read_ids(target, "activity_chembl_id", cfg, limit=1))
    assert ids == ["CHEMBL100"]


def test_chembl_client_handles_activity_404(
    requests_mock: requests_mock_lib.Mocker,
) -> None:
    base_url = "https://chembl.mock"
    requests_mock.get(f"{base_url}/activity/CHEMBL404.json", status_code=404)

    client = ChemblClient(base_url=base_url)
    assert client.fetch_activity("CHEMBL404") is None
    assert requests_mock.call_count == 1


def test_get_activities_batches_requests() -> None:
    calls: List[List[str]] = []

    class DummyClient:
        def fetch_many_activities(self, values: Iterable[str]) -> List[dict[str, str]]:
            batch = list(values)
            calls.append(batch)
            return [
                {
                    "activity_chembl_id": activity_id,
                    "assay_chembl_id": f"ASSAY-{activity_id}",
                }
                for activity_id in batch
            ]

    df = get_activities(DummyClient(), ["CHEMBL1", "CHEMBL2", "CHEMBL3"], chunk_size=2)
    assert list(df["activity_chembl_id"]) == ["CHEMBL1", "CHEMBL2", "CHEMBL3"]
    assert calls == [["CHEMBL1", "CHEMBL2"], ["CHEMBL3"]]


def test_normalize_activities() -> None:
    raw = pd.DataFrame(
        [
            {
                "activity_chembl_id": " ACT1 ",
                "assay_chembl_id": " ASSAY1 ",
                "molecule_chembl_id": " MOL1 ",
                "standard_value": "10",
                "standard_flag": "1",
                "potential_duplicate": "True",
                "ligand_efficiency": {"LE": 1, "LLE": 2},
                "activity_properties": [
                    {"name": "propB", "value": 2},
                    {"value": 1, "name": "propA"},
                ],
            },
            {
                "activity_chembl_id": "ACT2",
                "assay_chembl_id": "ASSAY2",
                "standard_value": None,
                "data_validity_warning": "false",
                "activity_properties": None,
            },
        ]
    )

    normalised = normalize_activities(raw)
    assert normalised.loc[0, "activity_chembl_id"] == "ACT1"
    assert normalised.loc[0, "assay_chembl_id"] == "ASSAY1"
    assert normalised.loc[0, "standard_value"] == pytest.approx(10.0)
    assert normalised.loc[0, "standard_flag"] == 1
    assert bool(normalised.loc[0, "potential_duplicate"]) is True
    assert normalised.loc[0, "ligand_efficiency"] == {"LE": 1, "LLE": 2}
    assert normalised.loc[0, "activity_properties"] == [
        {"name": "propA", "value": 1},
        {"name": "propB", "value": 2},
    ]
    assert pd.isna(normalised.loc[1, "standard_value"])
    assert bool(normalised.loc[1, "data_validity_warning"]) is False
    assert normalised.loc[1, "activity_properties"] == []


def test_validate_activities_writes_errors(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "activity_chembl_id": "CHEMBL1",
                "assay_chembl_id": "ASSAY1",
            },
            {
                "activity_chembl_id": "",
                "assay_chembl_id": None,
            },
        ]
    )

    errors_path = tmp_path / "errors.json"
    validated = validate_activities(df, errors_path=errors_path)

    assert len(validated) == 1
    assert not validated["activity_chembl_id"].isna().any()
    assert errors_path.exists()
    data = json.loads(errors_path.read_text(encoding="utf-8"))
    assert len(data) == 1


def test_validate_activities_handles_numpy_payloads(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "activity_chembl_id": "CHEMBL1",
                "assay_chembl_id": "ASSAY1",
                "activity_properties": np.array([], dtype=object),
                "target_components": np.array([{"name": "component"}], dtype=object),
                "standard_value": np.array([np.nan]),
            }
        ]
    )

    errors_path = tmp_path / "errors.json"
    validated = validate_activities(df, errors_path=errors_path)

    assert not errors_path.exists()
    assert validated.loc[0, "activity_properties"] == []
    assert validated.loc[0, "target_components"] == [{"name": "component"}]
    assert pd.isna(validated.loc[0, "standard_value"])


def test_chembl_activities_main_dry_run(tmp_path: Path) -> None:
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("activity_chembl_id\nCHEMBL1\nCHEMBL2\n", encoding="utf-8")

    output_csv = tmp_path / "out.csv"
    exit_code = chembl_activities_main(
        [
            "--input",
            str(input_csv),
            "--output",
            str(output_csv),
            "--dry-run",
            "--log-level",
            "DEBUG",
        ]
    )

    assert exit_code == 0
    assert not output_csv.exists()


def test_chembl_activities_main_end_to_end(
    tmp_path: Path, requests_mock: requests_mock_lib.Mocker
) -> None:
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("activity_chembl_id\nCHEMBL1\nCHEMBL2\n", encoding="utf-8")

    base_url = "https://chembl.mock"
    payload_template = {
        "activity_chembl_id": "",
        "assay_chembl_id": "ASSAY",
        "molecule_chembl_id": "MOL",
        "standard_value": 5,
        "standard_flag": 1,
        "potential_duplicate": False,
    }
    for activity_id in ["CHEMBL1", "CHEMBL2"]:
        payload = dict(payload_template)
        payload["activity_chembl_id"] = activity_id
        requests_mock.get(f"{base_url}/activity/{activity_id}.json", json=payload)

    output_csv = tmp_path / "out.csv"
    exit_code = chembl_activities_main(
        [
            "--input",
            str(input_csv),
            "--output",
            str(output_csv),
            "--base-url",
            base_url,
            "--chunk-size",
            "1",
            "--log-level",
            "DEBUG",
        ]
    )

    assert exit_code == 0
    assert output_csv.exists()
    df = pd.read_csv(output_csv)
    assert sorted(df["activity_chembl_id"].tolist()) == ["CHEMBL1", "CHEMBL2"]

    meta_file = output_csv.with_suffix(".csv.meta.yaml")
    assert meta_file.exists()

    quality_report = Path(f"{output_csv.with_suffix('')}_quality_report_table.csv")
    assert quality_report.exists()
    corr_report = Path(
        f"{output_csv.with_suffix('')}_data_correlation_report_table.csv"
    )
    assert corr_report.exists()
