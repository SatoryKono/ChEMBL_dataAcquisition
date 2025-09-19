from __future__ import annotations

import csv
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
import yaml

from library import io as library_io  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.get_target_data_main as get_target_data_main  # noqa: E402


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

    call_count = 0

    def counting_read_ids(
        path: Path, column: str, cfg: object, **kwargs: object
    ) -> Iterator[str]:
        nonlocal call_count
        call_count += 1
        return library_io.read_ids(path, column, cfg, **kwargs)

    monkeypatch.setattr(get_target_data_main, "read_ids", counting_read_ids)

    quality_calls: list[tuple[object, str, str, str]] = []

    def fake_quality(
        table: object, *, table_name: str, separator: str, encoding: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        quality_calls.append((table, table_name, separator, encoding))
        return pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(
        "scripts.get_target_data_main.analyze_table_quality", fake_quality
    )

    get_target_data_main.main(
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

    assert call_count == 1

    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert [row["target_chembl_id"] for row in rows] == ["CHEMBL123", "CHEMBL999"]
    assert rows[0]["pref_name"] == "Example target"
    cross_refs = json.loads(rows[0]["cross_references"])
    assert cross_refs == [{"xref_id": "P12345", "source": "UniProt"}]

    meta_path = output_csv.with_name(f"{output_csv.name}.meta.yaml")
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    assert meta["rows"] == 2
    assert meta["columns"] == 3
    assert meta["output"] == str(output_csv)

    assert quality_calls == [
        (output_csv, str(output_csv.with_suffix("")), ",", "utf-8")
    ]


def test_get_target_data_cli_streams_in_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the CLI processes identifiers in fixed-size batches."""

    input_csv = tmp_path / "targets.csv"
    input_csv.write_text(
        "target_chembl_id\nchembl001\nchembl002\nchembl003\nchembl004\n",
        encoding="utf-8",
    )
    output_csv = tmp_path / "dump.csv"

    calls: list[list[str]] = []

    def fake_fetch(ids: list[str], _cfg: object) -> pd.DataFrame:
        calls.append(list(ids))
        rows = [
            {
                "target_chembl_id": chembl_id,
                "pref_name": f"Target {chembl_id[-3:]}",
                "cross_references": [{"source": "UniProt", "xref_id": f"P{index:05d}"}],
            }
            for index, chembl_id in enumerate(ids, start=1)
        ]
        return pd.DataFrame(rows)

    monkeypatch.setattr("scripts.get_target_data_main.fetch_targets", fake_fetch)
    monkeypatch.setattr("scripts.get_target_data_main.STREAM_BATCH_SIZE", 2)

    def fake_quality(
        table: Path, *, table_name: str, separator: str, encoding: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        assert table == output_csv
        assert table_name == str(output_csv.with_suffix(""))
        assert separator == ","
        assert encoding == "utf-8"
        return pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(
        "scripts.get_target_data_main.analyze_table_quality", fake_quality
    )

    get_target_data_main.main(
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

    assert calls == [["CHEMBL001", "CHEMBL002"], ["CHEMBL003", "CHEMBL004"]]

    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert [row["target_chembl_id"] for row in rows] == [
        "CHEMBL001",
        "CHEMBL002",
        "CHEMBL003",
        "CHEMBL004",
    ]
    assert rows[0]["cross_references"] == json.dumps(
        [{"source": "UniProt", "xref_id": "P00001"}], separators=(",", ":")
    )

    meta_path = output_csv.with_name(f"{output_csv.name}.meta.yaml")
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    assert meta["rows"] == 4
    assert meta["columns"] == 3
    assert meta["output"] == str(output_csv)
