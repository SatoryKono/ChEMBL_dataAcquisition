from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ruff: noqa: E402
from scripts import chembl2uniprot_main
from scripts import get_cell_line_main
from scripts import get_target_data_main
import scripts.pipeline_targets_main as pipeline_main


def test_get_target_data_main_missing_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The target metadata CLI exits with code 1 when the input file is absent."""

    missing_input = tmp_path / "missing.csv"
    output_path = tmp_path / "out.csv"
    with pytest.raises(SystemExit) as excinfo:
        get_target_data_main.main(
            [
                "--input",
                str(missing_input),
                "--output",
                str(output_path),
            ]
        )
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "does not exist" in captured.err


def test_get_cell_line_main_missing_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The cell line CLI validates that the CSV file exists."""

    output_path = tmp_path / "cell_lines.json"
    with pytest.raises(SystemExit) as excinfo:
        get_cell_line_main.main(
            [
                "--input",
                str(tmp_path / "missing.csv"),
                "--output",
                str(output_path),
            ]
        )
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "does not exist" in captured.err


def test_pipeline_targets_main_missing_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The pipeline CLI reports a helpful error when the input CSV is missing."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text("pipeline: {}\n", encoding="utf-8")
    argv = [
        "pipeline_targets_main.py",
        "--input",
        str(tmp_path / "missing.csv"),
        "--output",
        str(tmp_path / "out.csv"),
        "--config",
        str(config_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        pipeline_main.main()
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Input file" in captured.err


def test_chembl2uniprot_main_missing_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ChEMBL to UniProt CLI exits with an informative error message."""

    with pytest.raises(SystemExit) as excinfo:
        chembl2uniprot_main.main(
            [
                "--input",
                str(tmp_path / "missing.csv"),
                "--output",
                str(tmp_path / "out.csv"),
            ]
        )
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "does not exist" in captured.err
