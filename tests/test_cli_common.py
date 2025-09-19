"""Unit tests for :mod:`library.cli_common`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from hypothesis import given, settings, strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ruff: noqa: E402
from library.cli_common import (
    ensure_output_dir,
    prepare_cli_config,
    resolve_cli_sidecar_paths,
    serialise_dataframe,
    write_cli_metadata,
)
from library.io_utils import serialise_cell

LIST_VALUES = st.lists(
    st.one_of(st.integers(-10, 10), st.text(min_size=1, max_size=5)),
    max_size=5,
)
DICT_VALUES = st.dictionaries(
    st.text(min_size=1, max_size=5),
    st.integers(-50, 50),
    max_size=5,
)
OBJECT_VALUES = st.one_of(
    LIST_VALUES,
    DICT_VALUES,
    st.text(min_size=0, max_size=10),
    st.integers(-100, 100),
    st.none(),
)


def _baseline_serialise(df: pd.DataFrame, list_format: str) -> pd.DataFrame:
    """Replicates the legacy :func:`serialise_dataframe` implementation."""

    result = df.copy()
    for col_name in result.columns:
        result[col_name] = result[col_name].map(
            lambda value: serialise_cell(value, list_format)
        )
    return result


def _time_execution(
    func: Callable[[], pd.DataFrame | None], *, repeats: int = 3
) -> float:
    """Measure the execution time of a callable and return the best duration."""

    durations: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        durations.append(time.perf_counter() - start)
    return min(durations) if durations else 0.0


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


def test_serialise_dataframe_supports_inplace_updates() -> None:
    """Setting ``inplace=True`` should mutate and return the original frame."""

    df = pd.DataFrame({"list_col": [[1, 2], [3, 4]], "value": [1, 2]})

    serialised = serialise_dataframe(df, "json", inplace=True)

    assert serialised is df
    assert serialised.loc[0, "list_col"] == "[1,2]"
    # Numeric columns are left untouched and retain their dtype.
    assert np.issubdtype(serialised["value"].dtype, np.integer)


@settings(max_examples=10, deadline=None)
@given(
    data_frames(
        columns=[
            column("lists", dtype=object, elements=LIST_VALUES),
            column("dicts", dtype=object, elements=DICT_VALUES),
            column("mixed", dtype=object, elements=OBJECT_VALUES),
            column(
                "strings",
                dtype=object,
                elements=st.text(min_size=0, max_size=15),
            ),
            column(
                "numbers",
                dtype=np.int64,
                elements=st.integers(-1_000, 1_000),
            ),
        ],
        index=range_indexes(min_size=32, max_size=128),
    )
)
def test_serialise_dataframe_property_equivalence(frame: pd.DataFrame) -> None:
    """Hypothesis-based regression test comparing against the legacy behaviour."""

    baseline = _baseline_serialise(frame, "pipe")
    serialised = serialise_dataframe(frame, "pipe")

    pd.testing.assert_frame_equal(serialised, baseline, check_dtype=False)


def test_serialise_dataframe_large_dataset_performance() -> None:
    """The optimised implementation should outperform the baseline on large data."""

    row_count = 10_000
    df = pd.DataFrame(
        {
            "dict_col": [{"idx": i, "value": i % 5} for i in range(row_count)],
            "list_col": [[i, i + 1, i + 2] for i in range(row_count)],
            "string_col": [f"value|{i}" for i in range(row_count)],
            "number_col": np.arange(row_count, dtype=np.int64),
        }
    )

    baseline = _baseline_serialise(df, "pipe")
    serialised = serialise_dataframe(df, "pipe")

    pd.testing.assert_frame_equal(serialised, baseline, check_dtype=False)

    baseline_time = _time_execution(lambda: _baseline_serialise(df, "pipe"), repeats=5)
    optimised_time = _time_execution(lambda: serialise_dataframe(df, "pipe"), repeats=5)

    assert optimised_time <= baseline_time * 1.1


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

    assert meta_file == output_path.with_name(f"{output_path.name}.meta.yaml")
    payload = yaml.safe_load(meta_file.read_text(encoding="utf-8"))
    assert payload["command"] == "chembl --flag 'value with space'"
    assert payload["config"]["input"] == str(tmp_path / "input.csv")
    assert "output" not in payload["config"]
    assert payload["rows"] == 1
    assert payload["columns"] == 1
    assert payload["status"] == "success"


def test_write_cli_metadata_includes_warnings(tmp_path: Path) -> None:
    """Warnings supplied to the metadata writer should be persisted."""

    output_path = tmp_path / "reports" / "dataset.csv"
    ensure_output_dir(output_path)
    output_path.write_text("col\nvalue\n", encoding="utf-8")

    namespace = argparse.Namespace(
        input=tmp_path / "input.csv",
        output=str(output_path),
        errors_output=None,
        meta_output=None,
        limit=3,
    )

    warnings = ["First warning", "Second warning"]
    meta_file = write_cli_metadata(
        output_path,
        row_count=1,
        column_count=1,
        namespace=namespace,
        warnings=warnings,
    )

    payload = yaml.safe_load(meta_file.read_text(encoding="utf-8"))
    assert payload["warnings"] == warnings


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


def test_write_cli_metadata_records_error(tmp_path: Path) -> None:
    """The metadata writer should capture error outcomes when requested."""

    output_path = tmp_path / "results.csv"
    namespace = argparse.Namespace(
        input=tmp_path / "input.csv",
        output=str(output_path),
        errors_output=None,
        meta_output=None,
    )

    message = "Input file missing required column"
    meta_file = write_cli_metadata(
        output_path,
        row_count=0,
        column_count=0,
        namespace=namespace,
        status="error",
        error=message,
    )

    payload = yaml.safe_load(meta_file.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert payload["error"] == message
    assert payload["sha256"] is None
    assert "determinism" not in payload


def test_resolve_cli_sidecar_paths_defaults(tmp_path: Path) -> None:
    """The helper should derive default sidecar locations safely."""

    output_path = tmp_path / "nested" / "dataset.tar.gz"
    meta_path, errors_path, quality_base = resolve_cli_sidecar_paths(output_path)

    assert meta_path == output_path.with_name("dataset.tar.gz.meta.yaml")
    assert errors_path == output_path.with_name("dataset.tar.gz.errors.json")
    assert quality_base == output_path.with_name("dataset.tar")

    # Ensure overrides are honoured and converted to Path instances.
    override_meta, override_errors, override_quality = resolve_cli_sidecar_paths(
        output_path,
        meta_output=tmp_path / "custom.yaml",
        errors_output=str(tmp_path / "errors.json"),
    )

    assert override_meta == tmp_path / "custom.yaml"
    assert override_errors == tmp_path / "errors.json"
    assert override_quality == quality_base

    plain_output = tmp_path / "dataset"
    plain_meta, plain_errors, plain_quality = resolve_cli_sidecar_paths(plain_output)
    assert plain_meta == plain_output.with_name("dataset.meta.yaml")
    assert plain_errors == plain_output.with_name("dataset.errors.json")
    assert plain_quality == plain_output
