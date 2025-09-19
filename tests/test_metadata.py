"""Unit tests for :mod:`library.metadata`."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from library.cli_common import resolve_cli_sidecar_paths
from library.metadata import write_meta_yaml


def test_write_meta_yaml_honours_custom_destination(metadata_csv: Path) -> None:
    """The metadata writer should accept an explicit output path."""

    meta_path = metadata_csv.with_name(f"{metadata_csv.name}.custom.meta.yaml")
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


def test_write_meta_yaml_default_destination(metadata_csv: Path) -> None:
    """Default sidecar naming should append ``.meta.yaml`` safely."""

    result = write_meta_yaml(
        metadata_csv,
        command="python script.py",  # minimal command
        config={},
        row_count=1,
        column_count=1,
    )

    expected_meta = metadata_csv.with_name(f"{metadata_csv.name}.meta.yaml")
    assert result == expected_meta
    assert expected_meta.exists()
    payload = yaml.safe_load(expected_meta.read_text(encoding="utf-8"))
    assert payload["output"] == str(metadata_csv)


def test_write_meta_yaml_without_meta_path_creates_adjacent_file(
    tmp_path: Path,
) -> None:
    """Calling the helper without ``meta_path`` writes a sidecar next to the CSV."""

    nested_csv = tmp_path / "nested" / "dataset.tar.gz"
    nested_csv.parent.mkdir(parents=True, exist_ok=True)
    nested_csv.write_text("col\nvalue\n", encoding="utf-8")

    expected_meta, _, _ = resolve_cli_sidecar_paths(nested_csv)
    result = write_meta_yaml(
        nested_csv,
        command="python script.py",  # minimal command
        config={},
        row_count=1,
        column_count=1,
    )

    assert result == expected_meta
    assert result.parent == nested_csv.parent
    assert result.exists()
    payload = yaml.safe_load(result.read_text(encoding="utf-8"))
    assert payload["output"] == str(nested_csv)


def test_write_meta_yaml_updates_determinism_on_repeat(metadata_csv: Path) -> None:
    """Repeated invocations should update determinism metadata deterministically."""

    first_meta = write_meta_yaml(
        metadata_csv,
        command="python script.py",  # minimal command
        config={},
        row_count=1,
        column_count=1,
    )
    first_payload = yaml.safe_load(first_meta.read_text(encoding="utf-8"))

    second_meta = write_meta_yaml(
        metadata_csv,
        command="python script.py",  # same command for determinism
        config={},
        row_count=1,
        column_count=1,
    )
    assert second_meta == first_meta

    payload = yaml.safe_load(second_meta.read_text(encoding="utf-8"))
    determinism = payload["determinism"]

    assert determinism["current_sha256"] == payload["sha256"]
    assert determinism["previous_sha256"] == first_payload["sha256"]
    assert determinism["baseline_sha256"] == first_payload["determinism"]["baseline_sha256"]
    assert determinism["matches_previous"] is True
    assert determinism["check_count"] >= 2
