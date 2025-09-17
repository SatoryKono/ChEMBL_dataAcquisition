from __future__ import annotations

import json

import pytest

import chembl_tissue_main


def test_main_fetches_single_identifier(tmp_path, requests_mock) -> None:
    output_path = tmp_path / "tissue.json"
    requests_mock.get(
        "https://example.org/tissue/CHEMBL3988026.json",
        json={"tissue_chembl_id": "CHEMBL3988026", "pref_name": "Cervix"},
    )
    exit_code = chembl_tissue_main.main(
        [
            "--chembl-id",
            "CHEMBL3988026",
            "--output",
            str(output_path),
            "--base-url",
            "https://example.org",
            "--log-level",
            "ERROR",
        ]
    )
    assert exit_code == 0
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data[0]["pref_name"] == "Cervix"


def test_main_reads_csv_and_skips_missing(tmp_path, requests_mock) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text(
        "tissue_chembl_id\nCHEMBL3988026\ninvalid\nCHEMBL9999999\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "result.json"
    requests_mock.get(
        "https://example.org/tissue/CHEMBL3988026.json",
        json={"tissue_chembl_id": "CHEMBL3988026", "pref_name": "Cervix"},
    )
    requests_mock.get("https://example.org/tissue/CHEMBL9999999.json", status_code=404)
    exit_code = chembl_tissue_main.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--base-url",
            "https://example.org",
            "--skip-missing",
            "--log-level",
            "ERROR",
        ]
    )
    assert exit_code == 0
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert [record["tissue_chembl_id"] for record in data] == ["CHEMBL3988026"]


def test_main_errors_when_column_missing(tmp_path) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text("other\nCHEMBL1\n", encoding="utf-8")
    exit_code = chembl_tissue_main.main(
        [
            "--input",
            str(input_path),
            "--base-url",
            "https://example.org",
            "--log-level",
            "ERROR",
        ]
    )
    assert exit_code == 1


@pytest.mark.parametrize("invalid_id", ["", "   ", "CHEMBLXYZ", None])
def test_main_rejects_invalid_identifier(tmp_path, invalid_id) -> None:
    args = [
        "--output",
        str(tmp_path / "out.json"),
        "--base-url",
        "https://example.org",
        "--log-level",
        "ERROR",
    ]
    if invalid_id is None:
        exit_code = chembl_tissue_main.main(args)
        assert exit_code == 1
    else:
        exit_code = chembl_tissue_main.main(
            [
                "--chembl-id",
                invalid_id,
                "--output",
                str(tmp_path / "out.json"),
                "--base-url",
                "https://example.org",
                "--log-level",
                "ERROR",
            ]
        )
        assert exit_code == 1
