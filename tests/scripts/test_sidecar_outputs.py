"""Regression tests for CLI sidecar files."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests_mock
import yaml

from library.cli_common import resolve_cli_sidecar_paths


@pytest.fixture(name="cell_line_module")
def fixture_cell_line_module() -> Any:
    """Return the ``get_cell_line_main`` module for patchable access."""

    return importlib.import_module("scripts.get_cell_line_main")


@pytest.fixture(name="hgnc_module")
def fixture_hgnc_module() -> Any:
    """Return the ``get_hgnc_by_uniprot`` module for patchable access."""

    return importlib.import_module("scripts.get_hgnc_by_uniprot")


def test_get_cell_line_cli_writes_sidecars(
    tmp_path: Path,
    requests_mock: requests_mock.Mocker,
    cell_line_module: Any,
) -> None:
    """Ensure the cell line CLI emits metadata and error reports."""

    output_path = tmp_path / "outputs" / "cell_lines.json"
    meta_output = tmp_path / "meta" / "cell_lines.meta.yaml"
    errors_output = tmp_path / "errors" / "cell_lines.errors.json"

    success_url = "https://example.org/cell_line/CHEMBL123.json"
    missing_url = "https://example.org/cell_line/CHEMBL404.json"
    requests_mock.get(success_url, json={"cell_chembl_id": "CHEMBL123"})
    requests_mock.get(missing_url, status_code=404)

    with pytest.raises(SystemExit) as exit_info:
        cell_line_module.main(
            [
                "--cell-line-id",
                "CHEMBL123",
                "--cell-line-id",
                "CHEMBL404",
                "--output",
                str(output_path),
                "--meta-output",
                str(meta_output),
                "--errors-output",
                str(errors_output),
                "--base-url",
                "https://example.org",
            ]
        )

    assert exit_info.value.code == 1

    assert meta_output.exists()
    metadata = yaml.safe_load(meta_output.read_text(encoding="utf-8"))
    assert metadata["rows"] == 1
    assert metadata["columns"] >= 1
    assert metadata["output"] == str(output_path)

    assert errors_output.exists()
    errors_payload = json.loads(errors_output.read_text(encoding="utf-8"))
    assert len(errors_payload) == 1
    assert errors_payload[0]["cell_line_id"] == "CHEMBL404"
    assert "CHEMBL404" in errors_payload[0]["error"]


def test_hgnc_cli_writes_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hgnc_module: Any,
) -> None:
    """Ensure the HGNC CLI emits metadata and error reports."""

    input_csv = tmp_path / "input.csv"
    input_csv.write_text("uniprot_id\nP12345\nP99999\n", encoding="utf-8")

    output_path = tmp_path / "results" / "mapped.csv"
    meta_output = tmp_path / "meta" / "mapped.meta.yaml"
    errors_output = tmp_path / "errors" / "mapped.errors.json"

    def fake_map_uniprot_to_hgnc(
        input_csv_path: Path,
        output_csv_path: Path | None,
        config_path: Path,
        *,
        config_section: str | None = None,
        column: str = "uniprot_id",
        sep: str | None = None,
        encoding: str | None = None,
        log_level: str = "INFO",
    ) -> Path:
        destination = output_csv_path or input_csv_path.with_name(
            f"hgnc_{input_csv_path.stem}.csv"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        dataframe = pd.DataFrame(
            [
                {
                    "uniprot_id": "P12345",
                    "hgnc_id": "HGNC:5",
                    "gene_symbol": "GENE1",
                    "gene_name": "Gene Name 1",
                    "protein_name": "Protein 1",
                },
                {
                    "uniprot_id": "P99999",
                    "hgnc_id": "",
                    "gene_symbol": "",
                    "gene_name": "",
                    "protein_name": "",
                },
            ]
        )
        delimiter = sep or ","
        file_encoding = encoding or "utf-8"
        dataframe.to_csv(destination, sep=delimiter, encoding=file_encoding, index=False)
        return destination

    monkeypatch.setattr(hgnc_module, "map_uniprot_to_hgnc", fake_map_uniprot_to_hgnc)

    hgnc_module.main(
        [
            "--input",
            str(input_csv),
            "--output",
            str(output_path),
            "--meta-output",
            str(meta_output),
            "--errors-output",
            str(errors_output),
            "--sep",
            ",",
            "--encoding",
            "utf-8",
        ]
    )

    assert meta_output.exists()
    metadata = yaml.safe_load(meta_output.read_text(encoding="utf-8"))
    assert metadata["rows"] == 2
    assert metadata["columns"] == 5
    assert metadata["output"] == str(output_path.resolve())

    assert errors_output.exists()
    errors_payload = json.loads(errors_output.read_text(encoding="utf-8"))
    assert errors_payload == [
        {"uniprot_id": "P99999", "error": "HGNC identifier missing"}
    ]


def test_resolve_cli_sidecar_paths_expands_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure custom sidecar destinations expand ``~`` to the home directory."""

    # ``Path.expanduser`` resolves ``~`` against the ``HOME`` or ``USERPROFILE``
    # environment variables. Setting both makes the behaviour deterministic
    # across operating systems.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    output_path = tmp_path / "results.csv"
    meta_override = "~/overrides/meta.yaml"
    errors_override = "~/overrides/errors.json"

    meta_path, errors_path, quality_base = resolve_cli_sidecar_paths(
        output_path,
        meta_output=meta_override,
        errors_output=errors_override,
    )

    expected_meta = tmp_path / "overrides" / "meta.yaml"
    expected_errors = tmp_path / "overrides" / "errors.json"

    assert meta_path == expected_meta
    assert errors_path == expected_errors
    assert quality_base == output_path.with_name(output_path.stem)
