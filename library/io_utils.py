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
    ids = [str(v).strip().upper() for v in df[column].fillna("")]
    seen = set()
    unique: List[str] = []
    for v in ids:
        if v and v not in seen:
            unique.append(v)
            seen.add(v)
    return unique


def _escape_pipe(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|")


def _serialise_list(values: Iterable[Any], list_format: str) -> str:
    def _normalise(v: Any) -> Any:
        if isinstance(v, tuple):
            # Represent domain tuples as id|name or JSON object
            if list_format == "json":
                return {"id": v[0], "name": v[1]}
            return f"{_escape_pipe(str(v[0]))}|{_escape_pipe(str(v[1]))}"
        return _escape_pipe(str(v)) if list_format == "pipe" else v

    norm = [_normalise(v) for v in values]
    if list_format == "pipe":
        return "|".join(str(v) for v in norm)
    if list_format == "json":
        return json.dumps(norm, separators=(",", ":"))
    raise ValueError(f"Unknown list_format: {list_format}")


def _serialise_value(value: Any, list_format: str) -> str:
    if isinstance(value, list):
        return _serialise_list(value, list_format)
    return str(value)


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
                [_serialise_value(row.get(col, ""), cfg.list_format) for col in columns]
            )
