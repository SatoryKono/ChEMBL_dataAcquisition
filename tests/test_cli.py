from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"
CONFIG_DIR = DATA_DIR / "config"
CSV_DIR = DATA_DIR / "csv"
ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "library"

CONFIG = CONFIG_DIR / "valid.yaml"
INVALID_CONFIG = CONFIG_DIR / "invalid.yaml"
EMPTY_CSV = CSV_DIR / "empty.csv"


def test_cli_runs(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "chembl2uniprot",
            "--input",
            str(EMPTY_CSV),
            "--output",
            str(out),
            "--config",
            str(CONFIG),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(LIB)},
    )
    assert out.exists()
    # Output should contain CSV header only
    assert out.read_text().strip() == "chembl_id,mapped_uniprot_id"
    # CLI prints output path
    assert result.stdout.strip() == str(out)


def test_cli_invalid_config(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "chembl2uniprot",
                "--input",
                str(EMPTY_CSV),
                "--output",
                str(out),
                "--config",
                str(INVALID_CONFIG),
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(LIB)},
        )
