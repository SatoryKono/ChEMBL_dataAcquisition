from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.get_target_data_main import main  # noqa: E402


def test_get_target_data_cli_writes_csv_and_meta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Smoke-test the target metadata CLI output and metadata sidecar."""

    input_csv = tmp_path / "targets.csv"
    input_csv.write_text(
        "target_chembl_id\nchembl123\nCHEMBL123\nchembl999\n",
        encoding="utf-8",
    )
    output_csv = tmp_path / "dump.csv"

    sample = pd.DataFrame(
        [
            {
                "target_chembl_id": "CHEMBL123",
                "pref_name": "Example target",
                "cross_references": [{"source": "UniProt", "xref_id": "P12345"}],
            },
            {
                "target_chembl_id": "CHEMBL999",
                "pref_name": "Fallback",
                "cross_references": [],
            },
        ]
    )

    def fake_fetch(ids: list[str], _cfg: object) -> pd.DataFrame:
        assert ids == ["CHEMBL123", "CHEMBL999"]
        return sample

    monkeypatch.setattr("scripts.get_target_data_main.fetch_targets", fake_fetch)

    main(
        [
            "--input",
            str(input_csv),
            "--output",
            str(output_csv),
            "--column",
            "target_chembl_id",
            "--encoding",
            "utf-8",
            "--sep",
            ",",
            "--list-format",
            "json",
        ]
    )

    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert [row["target_chembl_id"] for row in rows] == ["CHEMBL123", "CHEMBL999"]
    assert rows[0]["pref_name"] == "Example target"
    cross_refs = json.loads(rows[0]["cross_references"])
    assert cross_refs == [{"xref_id": "P12345", "source": "UniProt"}]

    meta_path = output_csv.with_suffix(f"{output_csv.suffix}.meta.yaml")
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    assert meta["rows"] == 2
    assert meta["columns"] == 3
    assert meta["output"] == str(output_csv)
