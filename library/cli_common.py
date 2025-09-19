"""Shared utilities for command-line interfaces."""

from __future__ import annotations

import shlex
import sys
from argparse import Namespace
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from library.data_profiling import analyze_table_quality
from library.io_utils import serialise_cell
from library.metadata import write_meta_yaml

__all__ = [
    "ensure_output_dir",
    "serialise_dataframe",
    "prepare_cli_config",
    "write_cli_metadata",
    "resolve_cli_sidecar_paths",
    "analyze_table_quality",
]

_EXCLUDED_CONFIG_KEYS = frozenset({"output", "errors_output", "meta_output"})


def _append_suffix_to_name(path: Path, suffix: str) -> Path:
    """Return ``path`` with ``suffix`` appended to the filename.

    Args:
        path: The base file path used to derive the companion file.
        suffix: The suffix to append, typically starting with a dot (e.g.
            ``".meta.yaml"``).

    Returns:
        A new :class:`pathlib.Path` instance that appends ``suffix`` to the
        original filename while preserving the parent directory.

    Raises:
        ValueError: If ``suffix`` is empty or ``path`` lacks a filename.
    """

    if not suffix:
        msg = "suffix must be a non-empty string"
        raise ValueError(msg)
    if path.name == "":
        msg = "path must include a filename"
        raise ValueError(msg)
    return path.with_name(f"{path.name}{suffix}")


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


ListFormat = Literal["json", "pipe"]


def serialise_dataframe(
    df: pd.DataFrame, list_format: ListFormat, *, inplace: bool = False
) -> pd.DataFrame:
    """Serializes non-scalar DataFrame columns for CSV output.

    This helper mirrors :func:`library.io_utils.serialise_cell` but operates on
    complete :class:`pandas.DataFrame` instances. Only object-like columns are
    materialised, which keeps numeric and boolean columns as zero-copy views of
    the original frame. When working with very large tables the caller can set
    ``inplace=True`` to update the provided DataFrame directly and avoid even the
    shallow copy that backs the default behaviour.

    Parameters
    ----------
    df:
        The DataFrame containing data destined for CSV output.
    list_format:
        The desired representation for list-like values. Accepted values are
        ``"json"`` and ``"pipe"``.
    inplace:
        Mutate ``df`` directly when set to :data:`True`. This avoids creating a
        new DataFrame instance and is therefore preferred when the original
        object is no longer needed prior to serialization.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the same shape as the input, where complex values are
        serialized into deterministic strings. The result is ``df`` itself when
        ``inplace=True``.

    Raises
    ------
    ValueError
        If an unsupported ``list_format`` is provided.

    Notes
    -----
    The function operates column by column to minimise memory pressure. Only
    object, string, and categorical columns are materialised; numeric columns
    retain their original dtype and continue to share buffers with ``df``. When
    chaining with :meth:`pandas.DataFrame.to_csv` the overall memory usage still
    scales with the size of the DataFrame because pandas keeps the table in
    memory until the CSV export completes. Consider chunked writes or
    ``inplace=True`` when handling multi-gigabyte datasets.
    """

    if list_format not in {"json", "pipe"}:
        msg = f"Unsupported list_format: {list_format}"
        raise ValueError(msg)

    serialised_frame = df if inplace else df.copy(deep=False)

    object_like = serialised_frame.select_dtypes(
        include=["object", "string", "category"]
    )
    if object_like.empty:
        return serialised_frame

    def _serialise_column(column: pd.Series) -> pd.Series:
        return column.map(lambda value: serialise_cell(value, list_format))

    transformed = object_like.apply(_serialise_column)
    for column_name, column_values in transformed.items():
        serialised_frame[column_name] = column_values

    return serialised_frame


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


def resolve_cli_sidecar_paths(
    output_path: Path,
    *,
    meta_output: str | Path | None = None,
    errors_output: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Derive companion file locations for CLI commands.

    The helper centralises the logic for computing metadata, validation error
    and quality-report destinations.  It ensures filenames are constructed by
    appending the relevant suffixes to the *full* output filename, which is
    important for multi-extension outputs such as ``.tar.gz``.

    Args:
        output_path: Path to the primary CSV or TSV file written by the CLI.
        meta_output: Optional override for the metadata destination provided by
            the user via ``--meta-output``. When omitted the helper appends
            ``".meta.yaml"`` to the output filename.
        errors_output: Optional override for the validation errors report. When
            omitted the helper appends ``".errors.json"`` to the output
            filename.

    Returns:
        A tuple ``(meta_path, errors_path, quality_base)`` where ``meta_path``
        and ``errors_path`` are concrete :class:`pathlib.Path` instances and
        ``quality_base`` is a path without the final suffix that can be passed
        to :func:`library.data_profiling.analyze_table_quality`.

    Raises:
        ValueError: If ``output_path`` does not include a filename.
    """

    destination = Path(output_path)
    if destination.name == "":
        msg = "output_path must include a filename"
        raise ValueError(msg)

    meta_path = (
        Path(meta_output).expanduser()
        if meta_output is not None
        else _append_suffix_to_name(destination, ".meta.yaml")
    )
    errors_path = (
        Path(errors_output).expanduser()
        if errors_output is not None
        else _append_suffix_to_name(destination, ".errors.json")
    )
    if destination.suffix:
        quality_base = destination.with_name(destination.stem)
    else:
        quality_base = destination
    return meta_path, errors_path, quality_base


def write_cli_metadata(
    output_path: Path,
    *,
    row_count: int,
    column_count: int,
    namespace: Namespace | Mapping[str, Any],
    command_parts: Sequence[str] | None = None,
    meta_path: Path | None = None,
    status: Literal["success", "error"] = "success",
    error: str | None = None,
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
        status: Execution outcome recorded in the metadata sidecar. ``"success"``
            denotes a completed run, while ``"error"`` captures failures that
            prevented the output from being written.
        error: Optional human-readable description of the failure when
            ``status`` is ``"error"``.

    Returns:
        The path to the written metadata file.
    """

    command_sequence = command_parts if command_parts is not None else tuple(sys.argv)
    command = " ".join(shlex.quote(part) for part in command_sequence)
    config = prepare_cli_config(namespace)
    include_hash = status == "success"
    return write_meta_yaml(
        Path(output_path),
        command=command,
        config=config,
        row_count=row_count,
        column_count=column_count,
        meta_path=meta_path,
        status=status,
        error=error,
        include_hash=include_hash,
    )
