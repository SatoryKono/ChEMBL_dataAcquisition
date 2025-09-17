"""Deterministic CSV input/output helpers."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd


@dataclass
class CsvConfig:
    sep: str
    encoding: str
    list_format: str = "json"


def read_ids(path: Path, column: str, cfg: CsvConfig) -> List[str]:
    """Read a CSV file and return normalised, deduplicated accession IDs."""

    df = pd.read_csv(path, sep=cfg.sep, encoding=cfg.encoding, dtype=str)
    if column not in df.columns:
        msg = f"Missing required column '{column}'"
        raise KeyError(msg)
    ids = [str(v).strip().upper() for v in df[column].fillna("")]  # type: ignore[arg-type]
    seen = set()
    unique: List[str] = []
    for v in ids:
        if v and v not in seen:
            unique.append(v)
            seen.add(v)
    return unique


def _escape_pipe(value: str) -> str:
    """Escape pipe characters in a string.

    Parameters
    ----------
    value:
        The string to escape.

    Returns
    -------
    str
        The string with pipe characters escaped.
    """

    return value.replace("|", "\\|")


def _serialise_list(values: Iterable[Any], list_format: str) -> str:
    """Serialize an iterable of values into a string.

    The serialization format is determined by `list_format`.

    Parameters
    ----------
    values:
        An iterable of values to serialize.
    list_format:
        The format to use for serialization ("pipe" or "json").

    Returns
    -------
    str
        The serialized string.
    """

    def _normalise(v: Any) -> Any:
        if isinstance(v, tuple):
            # Represent domain tuples as id|name or JSON object
            if list_format == "json":
                return {"id": v[0], "name": v[1]}
            return f"{_escape_pipe(str(v[0]))}|{_escape_pipe(str(v[1]))}"
        if isinstance(v, dict):
            # Sort keys for determinism prior to JSON serialisation
            ordered = {key: v[key] for key in sorted(v)}
            if list_format == "json":
                return ordered
            return _escape_pipe(
                json.dumps(ordered, ensure_ascii=False, sort_keys=False)
            )
        if isinstance(v, list):
            if list_format == "json":
                return [_normalise(item) for item in v]
            return _escape_pipe(_serialise_list(v, list_format))
        return _escape_pipe(str(v)) if list_format == "pipe" else v

    norm = [_normalise(v) for v in values]
    if list_format == "pipe":
        return "|".join(str(v) for v in norm)
    if list_format == "json":
        return json.dumps(norm, separators=(",", ":"))
    raise ValueError(f"Unknown list_format: {list_format}")


def _serialise_value(value: Any, list_format: str) -> str:
    """Serialize a single value to a string.

    If the value is a list, it is serialized using `_serialise_list`.
    Otherwise, it is converted to a string.

    Parameters
    ----------
    value:
        The value to serialize.
    list_format:
        The format to use for lists ("pipe" or "json").

    Returns
    -------
    str
        The serialized string.
    """
    if isinstance(value, dict):
        ordered = {key: value[key] for key in sorted(value)}
        text = json.dumps(ordered, ensure_ascii=False, sort_keys=False)
        return _escape_pipe(text) if list_format == "pipe" else text
    if isinstance(value, list):
        return _serialise_list(value, list_format)
    return str(value)


def serialise_cell(value: Any, list_format: str) -> Any:
    """Serialise a value for CSV output while preserving separators.

    Parameters
    ----------
    value:
        The value to serialise. Lists and dictionaries are converted into
        deterministic JSON payloads using :func:`_serialise_list` and
        :func:`_serialise_value` respectively.
    list_format:
        Encoding for list-like values (``"json"`` or ``"pipe"``).

    Returns
    -------
    Any
        Serialised representation suitable for writing to CSV cells.
        Non-collection objects are returned unchanged except for strings in
        ``"pipe"`` mode where pipe characters are escaped.
    """

    if isinstance(value, (dict, list)):
        return _serialise_value(value, list_format)
    if isinstance(value, str) and list_format == "pipe":
        return _escape_pipe(value)
    return value


def write_rows(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    columns: Sequence[str],
    cfg: CsvConfig,
) -> None:
    """Write ``rows`` to ``path`` using deterministic ordering."""

    with path.open("w", encoding=cfg.encoding, newline="") as fh:
        writer = csv.writer(fh, delimiter=cfg.sep)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(
                [serialise_cell(row.get(col, ""), cfg.list_format) for col in columns]
            )
