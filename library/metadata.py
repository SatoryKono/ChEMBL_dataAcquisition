"""Helpers for writing dataset metadata sidecar files."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

LOGGER = logging.getLogger(__name__)


def _file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def write_meta_yaml(
    output_path: Path,
    *,
    command: str,
    config: Mapping[str, Any],
    row_count: int,
    column_count: int,
    meta_path: Path | None = None,
) -> Path:
    """Write dataset metadata next to ``output_path``.

    Parameters
    ----------
    output_path:
        Path to the generated CSV file.
    command:
        Command line invocation responsible for generating the data.
    config:
        Normalised configuration dictionary captured from the CLI arguments.
    row_count:
        Number of rows persisted to the CSV file.
    column_count:
        Number of columns in the CSV file.
    meta_path:
        Optional destination for the YAML file.  When omitted, the function
        writes ``<output_path>.meta.yaml``.
    """

    target = meta_path or output_path.with_suffix(f"{output_path.suffix}.meta.yaml")
    target.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "config": dict(config),
        "output": str(output_path),
        "sha256": _file_sha256(output_path),
        "rows": row_count,
        "columns": column_count,
    }

    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, allow_unicode=True, sort_keys=False)

    LOGGER.info("Metadata written to %s", target)
    return target
