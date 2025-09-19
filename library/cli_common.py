"""Shared utilities for command-line interfaces."""

from __future__ import annotations

import shlex
import sys
from argparse import Namespace
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from library.data_profiling import analyze_table_quality
from library.io_utils import serialise_cell
from library.metadata import write_meta_yaml

__all__ = [
    "ensure_output_dir",
    "serialise_dataframe",
    "prepare_cli_config",
    "write_cli_metadata",
    "analyze_table_quality",
]

_EXCLUDED_CONFIG_KEYS = frozenset({"output", "errors_output", "meta_output"})


def ensure_output_dir(path: Path) -> Path:
    """Ensures that the parent directory of a given path exists.

    Args:
        path: The destination file path.

    Returns:
        The normalized Path object pointing to the destination file.

    Raises:
        ValueError: If the path does not reference a file name.
    """

    target = Path(path)
    if target.name == "":
        msg = "Output path must include a file name"
        raise ValueError(msg)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def serialise_dataframe(df: pd.DataFrame, list_format: str) -> pd.DataFrame:
    """Serializes non-scalar DataFrame columns for CSV output.

    Args:
        df: The DataFrame containing data destined for CSV output.
        list_format: The desired representation for list-like values. Accepted
            values are "json" and "pipe".

    Returns:
        A new DataFrame with the same shape as the input, where complex values
        are serialized into deterministic strings.
    """

    result = df.copy()
    for column in result.columns:
        result[column] = result[column].map(
            lambda value: serialise_cell(value, list_format)
        )
    return result


def prepare_cli_config(
    namespace: Namespace | Mapping[str, Any],
    *,
    exclude_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Normalizes CLI arguments into a serializable mapping.

    Args:
        namespace: The source of configuration values, typically an
            `argparse.Namespace`, but any mapping is accepted.
        exclude_keys: An iterable of keys to omit from the result. If omitted,
            standard output-related keys are ignored.

    Returns:
        A dictionary representation of the namespace, with Path values
        converted to strings for YAML serialization.
    """

    excluded = set(_EXCLUDED_CONFIG_KEYS if exclude_keys is None else exclude_keys)
    items = (
        namespace.items() if isinstance(namespace, Mapping) else vars(namespace).items()
    )

    normalised: dict[str, Any] = {}
    for key, value in items:
        if key in excluded:
            continue
        normalised[key] = str(value) if isinstance(value, Path) else value
    return normalised


def write_cli_metadata(
    output_path: Path,
    *,
    row_count: int,
    column_count: int,
    namespace: Namespace | Mapping[str, Any],
    command_parts: Sequence[str] | None = None,
    meta_path: Path | None = None,
) -> Path:
    """Persists a `.meta.yaml` companion file next to the output file.

    Args:
        output_path: The path to the CSV file produced by the CLI.
        row_count: The number of rows in the output CSV.
        column_count: The number of columns in the output CSV.
        namespace: The CLI arguments used to produce the dataset, converted to a
            serializable mapping via `prepare_cli_config`.
        command_parts: A sequence of command-line arguments used to invoke the CLI.
            If omitted, the function falls back to `sys.argv`.
        meta_path: An optional override for the metadata file location. Defaults to
            `<output_path>.meta.yaml`.

    Returns:
        The path to the written metadata file.
    """

    command_sequence = command_parts if command_parts is not None else tuple(sys.argv)
    command = " ".join(shlex.quote(part) for part in command_sequence)
    config = prepare_cli_config(namespace)
    return write_meta_yaml(
        Path(output_path),
        command=command,
        config=config,
        row_count=row_count,
        column_count=column_count,
        meta_path=meta_path,
    )
