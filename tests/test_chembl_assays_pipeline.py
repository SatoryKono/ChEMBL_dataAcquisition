from __future__ import annotations

import json
import importlib
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import pytest
import requests_mock as requests_mock_lib
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

assay_postprocessing = importlib.import_module("library.assay_postprocessing")
normalize_assays = importlib.import_module("library.normalize_assays")
validate_assays = importlib.import_module("library.assay_validation").validate_assays
ChemblClient = importlib.import_module("library.chembl_client").ChemblClient
get_assays = importlib.import_module("library.chembl_library").get_assays
read_ids = importlib.import_module("library.io").read_ids
CsvConfig = importlib.import_module("library.io_utils").CsvConfig
write_meta_yaml = importlib.import_module("library.metadata").write_meta_yaml
chembl_assays_main = importlib.import_module("scripts.chembl_assays_main").main


def test_read_ids_streams_unique(tmp_path: Path) -> None:
    source = Path(__file__).parent / "data" / "assays_input.csv"
    target = tmp_path / "ids.csv"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    cfg = CsvConfig(sep=",", encoding="utf-8")
    ids = list(read_ids(target, "assay_chembl_id", cfg))
    assert ids == ["CHEMBL1", "CHEMBL2"]


def test_chembl_client_handles_404(requests_mock: requests_mock_lib.Mocker) -> None:
    base_url = "https://chembl.mock"
    requests_mock.get(f"{base_url}/assay/CHEMBL404.json", status_code=404)

    client = ChemblClient(base_url=base_url)
    assert client.fetch_assay("CHEMBL404") is None


def test_get_assays_batches_requests() -> None:
    calls: List[List[str]] = []

    class DummyClient:
        def fetch_many(self, values: Iterable[str]) -> List[dict[str, str]]:
            batch = list(values)
            calls.append(batch)
            return [
                {
                    "assay_chembl_id": assay_id,
                    "document_chembl_id": f"DOC-{assay_id}",
                    "target_chembl_id": f"TAR-{assay_id}",
                }
                for assay_id in batch
            ]

    df = get_assays(DummyClient(), ["CHEMBL1", "CHEMBL2", "CHEMBL3"], chunk_size=2)
    assert list(df["assay_chembl_id"]) == ["CHEMBL1", "CHEMBL2", "CHEMBL3"]
    assert calls == [["CHEMBL1", "CHEMBL2"], ["CHEMBL3"]]


def test_postprocess_and_normalize() -> None:
    raw = pd.DataFrame(
        [
            {
                "assay_chembl_id": "CHEMBL1",
                "document_chembl_id": " DOC1 ",
                "target_chembl_id": "TAR1",
                "confidence_score": "5",
                "assay_parameters": [
                    {"name": "ParamB", "value": 1},
                    {"value": 2, "name": "ParamA"},
                ],
            },
            {
                "assay_chembl_id": "CHEMBL2",
                "document_chembl_id": "DOC1",
                "target_chembl_id": "TAR1",
                "confidence_score": "",
                "assay_parameters": None,
            },
        ]
    )

    processed = assay_postprocessing.postprocess_assays(raw)
    assert list(processed["assay_with_same_target"]) == [2, 2]

    normalised = normalize_assays.normalize_assays(processed)
    assert normalised.loc[0, "document_chembl_id"] == "DOC1"
    assert normalised.loc[0, "assay_with_same_target"] == 2
    assert list(normalised.loc[0, "assay_parameters"]) == [
        {"name": "ParamA", "value": 2},
        {"name": "ParamB", "value": 1},
    ]
    assert pd.isna(normalised.loc[1, "confidence_score"])


def test_validate_assays_writes_errors(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "assay_chembl_id": "CHEMBL1",
                "document_chembl_id": "DOC1",
                "target_chembl_id": "TAR1",
                "assay_with_same_target": 1,
            },
            {
                "assay_chembl_id": "",
                "document_chembl_id": "DOC2",
                "target_chembl_id": "TAR2",
                "assay_with_same_target": -1,
            },
        ]
    )

    errors_path = tmp_path / "errors.json"
    validated = validate_assays(df, errors_path=errors_path)

    assert len(validated) == 1
    assert not validated["assay_chembl_id"].isna().any()
    assert errors_path.exists()
    data = json.loads(errors_path.read_text(encoding="utf-8"))
    assert len(data) == 1


def test_validate_assays_handles_numpy_payloads(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "assay_chembl_id": "CHEMBL1",
                "document_chembl_id": "DOC1",
                "target_chembl_id": "TAR1",
                "assay_with_same_target": 1,
                "assay_parameters": np.array([], dtype=object),
                "confidence_description": np.array([np.nan]),
            }
        ]
    )

    errors_path = tmp_path / "errors.json"
    validated = validate_assays(df, errors_path=errors_path)

    assert not errors_path.exists()
    assert validated.loc[0, "assay_parameters"] == []
    assert pd.isna(validated.loc[0, "confidence_description"])


def test_write_meta_yaml(tmp_path: Path) -> None:
    output = tmp_path / "assays.csv"
    output.write_text("assay_chembl_id\nCHEMBL1\n", encoding="utf-8")
    meta_path = write_meta_yaml(
        output,
        command="python script.py",
        config={"chunk_size": 10},
        row_count=1,
        column_count=1,
    )

    assert meta_path.exists()
    metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    assert metadata["rows"] == 1
    assert metadata["columns"] == 1
    assert "sha256" in metadata


def test_chembl_assays_main_end_to_end(
    tmp_path: Path, requests_mock: pytest.LogCaptureFixture
) -> None:
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("assay_chembl_id\nCHEMBL1\nCHEMBL2\n", encoding="utf-8")

    base_url = "https://chembl.mock"
    payload_template = {
        "assay_category": "Binding",
        "assay_chembl_id": "",
        "assay_group": "Group",
        "assay_type": "B",
        "assay_organism": "Human",
        "assay_test_type": "IC50",
        "document_chembl_id": "DOC",
        "target_chembl_id": "TAR",
        "confidence_score": 5,
        "assay_classifications": [{"level": 1, "label": "Test"}],
    }
    for assay_id in ["CHEMBL1", "CHEMBL2"]:
        payload = dict(payload_template)
        payload["assay_chembl_id"] = assay_id
        requests_mock.get(f"{base_url}/assay/{assay_id}.json", json=payload)

    output_csv = tmp_path / "out.csv"
    exit_code = chembl_assays_main(
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
    assert sorted(df["assay_chembl_id"].tolist()) == ["CHEMBL1", "CHEMBL2"]

    meta_file = output_csv.with_name(f"{output_csv.name}.meta.yaml")
    assert meta_file.exists()

    base_path = output_csv.with_name(output_csv.stem)
    quality_report = Path(f"{base_path}_quality_report_table.csv")
    assert quality_report.exists()
    corr_report = Path(
        f"{base_path}_data_correlation_report_table.csv"
    )
    assert corr_report.exists()
