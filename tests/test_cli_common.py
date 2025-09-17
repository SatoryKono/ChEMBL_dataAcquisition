"""Unit tests for :mod:`library.cli_common`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ruff: noqa: E402
from library.cli_common import (
    ensure_output_dir,
    prepare_cli_config,
    serialise_dataframe,
    write_cli_metadata,
)


def test_ensure_output_dir_creates_parent(tmp_path: Path) -> None:
    """The helper should create intermediate directories when necessary."""

    destination = tmp_path / "nested" / "output.csv"
    result = ensure_output_dir(destination)

    assert destination.parent.exists()
    assert result == destination


def test_ensure_output_dir_requires_filename() -> None:
    """An explicit file name is required to avoid ambiguous destinations."""

    with pytest.raises(ValueError):
        ensure_output_dir(Path(""))


def test_serialise_dataframe_preserves_original() -> None:
    """Non-scalar values should be serialised without mutating the input."""

    df = pd.DataFrame(
        {
            "dict_col": [{"x": 1}, {"y": 2}],
            "list_col": [[1, 2], ["pipe", "value"]],
            "str_col": ["pipe|value", "plain"],
        }
    )

    serialised = serialise_dataframe(df, "pipe")

    assert serialised is not df
    assert df.loc[0, "dict_col"] == {"x": 1}
    assert json.loads(serialised.loc[0, "dict_col"]) == {"x": 1}
    assert serialised.loc[0, "list_col"] == "1|2"
    assert serialised.loc[0, "str_col"] == "pipe\\|value"


def test_prepare_cli_config_normalises_paths(tmp_path: Path) -> None:
    """Configuration extraction should serialise paths and drop output keys."""

    namespace = argparse.Namespace(
        input=tmp_path / "input.csv",
        output=tmp_path / "out.csv",
        errors_output=None,
        meta_output=None,
        limit=5,
    )

    config = prepare_cli_config(namespace)

    assert "output" not in config
    assert "errors_output" not in config
    assert "meta_output" not in config
    assert config["input"] == str(tmp_path / "input.csv")
    assert config["limit"] == 5


def test_write_cli_metadata_produces_expected_yaml(tmp_path: Path) -> None:
    """Metadata writer should honour command quoting and config normalisation."""

    output_path = tmp_path / "reports" / "dataset.csv"
    ensure_output_dir(output_path)
    output_path.write_text("col\nvalue\n", encoding="utf-8")

    namespace = argparse.Namespace(
        input=tmp_path / "input.csv",
        output=str(output_path),
        errors_output=None,
        meta_output=None,
        limit=7,
    )

    meta_file = write_cli_metadata(
        output_path,
        row_count=1,
        column_count=1,
        namespace=namespace,
        command_parts=["chembl", "--flag", "value with space"],
    )

    assert meta_file == output_path.with_suffix(".csv.meta.yaml")
    payload = yaml.safe_load(meta_file.read_text(encoding="utf-8"))
    assert payload["command"] == "chembl --flag 'value with space'"
    assert payload["config"]["input"] == str(tmp_path / "input.csv")
    assert "output" not in payload["config"]
    assert payload["rows"] == 1
    assert payload["columns"] == 1
 


def test_write_cli_metadata_defaults_to_sys_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When no command parts are provided the helper should use ``sys.argv``."""

    output_path = tmp_path / "dataset.csv"
    ensure_output_dir(output_path)
    output_path.write_text("col\nvalue\n", encoding="utf-8")

    namespace = argparse.Namespace(
        input=tmp_path / "input.csv",
        output=str(output_path),
        errors_output=None,
        meta_output=None,
        limit=3,
    )

    monkeypatch.setattr(sys, "argv", ["chembl-cli", "--flag", "value"])
    meta_file = write_cli_metadata(
        output_path,
        row_count=1,
        column_count=1,
        namespace=namespace,
    )

    payload = yaml.safe_load(meta_file.read_text(encoding="utf-8"))
    assert payload["command"] == "chembl-cli --flag value"
 
    determinism = payload["determinism"]
    assert determinism["baseline_sha256"] == payload["sha256"]
    assert determinism["previous_sha256"] is None
    assert determinism["matches_previous"] is None
    assert determinism["check_count"] == 1
 
