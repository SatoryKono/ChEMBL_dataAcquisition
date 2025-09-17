"""Unit tests for :mod:`library.metadata`."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from library.metadata import write_meta_yaml


def test_write_meta_yaml_honours_custom_destination(metadata_csv: Path) -> None:
    """The metadata writer should accept an explicit output path."""

    meta_path = metadata_csv.with_suffix(".custom.meta.yaml")
    result = write_meta_yaml(
        metadata_csv,
        command="python script.py --flag",
        config={"chunk_size": 25},
        row_count=1,
        column_count=1,
        meta_path=meta_path,
    )

    assert result == meta_path
    assert meta_path.exists()
    payload = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(metadata_csv.read_bytes()).hexdigest()
    assert payload["output"] == str(metadata_csv)
    assert payload["command"] == "python script.py --flag"
    assert payload["config"]["chunk_size"] == 25
    assert payload["rows"] == 1
    assert payload["columns"] == 1
    assert payload["sha256"] == expected_hash
