from __future__ import annotations

import json
from pathlib import Path
import sys

import requests_mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.get_cell_line_main import main  # noqa: E402


def test_cli_fetch_single_id(
    tmp_path: Path, requests_mock: requests_mock.Mocker
) -> None:
    output = tmp_path / "cell_lines.json"
    url = "https://example.org/cell_line/CHEMBL123.json"
    requests_mock.get(url, json={"cell_chembl_id": "CHEMBL123"})
    main(
        [
            "--cell-line-id",
            "CHEMBL123",
            "--output",
            str(output),
            "--base-url",
            "https://example.org",
        ]
    )
    lines = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert lines == [{"cell_chembl_id": "CHEMBL123"}]


def test_cli_reads_ids_from_csv(
    tmp_path: Path, requests_mock: requests_mock.Mocker
) -> None:
    csv_path = tmp_path / "ids.csv"
    csv_path.write_text("cell_chembl_id\nCHEMBL777\n", encoding="utf-8")
    output = tmp_path / "out.json"
    url = "https://example.org/cell_line/CHEMBL777.json"
    requests_mock.get(url, json={"cell_chembl_id": "CHEMBL777"})
    main(
        [
            "--input",
            str(csv_path),
            "--output",
            str(output),
            "--base-url",
            "https://example.org",
        ]
    )
    data = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert data == [{"cell_chembl_id": "CHEMBL777"}]
