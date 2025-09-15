from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from uniprot_enrich import enrich_uniprot

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "uniprot_enrich_main.py"
DATA_FILE = Path(__file__).parent / "data" / "uniprot_sample.csv"


def test_uniprot_cli(tmp_path: Path) -> None:
    """CLI should enrich input and write to provided output path."""

    inp = tmp_path / "input.csv"
    inp.write_text(DATA_FILE.read_text())
    out = tmp_path / "out.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(inp),
            "--output",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == str(out)
    assert out.exists()
    df = pd.read_csv(out, dtype=str).fillna("")
    expected_cols = ["uniprot_id", "other"] + list(
        enrich_uniprot.__globals__["OUTPUT_COLUMNS"]
    )
    assert list(df.columns) == expected_cols
