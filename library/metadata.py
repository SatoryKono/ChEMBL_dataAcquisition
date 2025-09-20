"""Helpers for writing dataset metadata sidecar files."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]

LOGGER = logging.getLogger(__name__)


def _file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 digest for ``path`` using streamed reads."""

    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def _load_previous_metadata(path: Path) -> dict[str, Any]:
    """Load metadata from ``path`` if it exists and is well-formed."""

    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}
    except yaml.YAMLError as exc:  # type: ignore[attr-defined]
        LOGGER.warning("Failed to parse existing metadata %s: %s", path, exc)
        return {}
    if isinstance(payload, Mapping):
        return dict(payload)
    LOGGER.warning("Existing metadata in %s is not a mapping; ignoring", path)
    return {}


def _determinism_record(
    *, current_sha: str, previous_metadata: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Create a determinism summary for inclusion in metadata."""

    now = datetime.now(timezone.utc).isoformat()
    baseline_sha: str | None = None
    previous_sha: str | None = None
    matches_previous: bool | None = None
    check_count = 1

    if previous_metadata:
        previous_det = previous_metadata.get("determinism")
        if isinstance(previous_det, Mapping):
            baseline_value = previous_det.get("baseline_sha256")
            if isinstance(baseline_value, str):
                baseline_sha = baseline_value
            prev_value = previous_det.get("current_sha256")
            if isinstance(prev_value, str):
                previous_sha = prev_value
            prev_count = previous_det.get("check_count")
            if isinstance(prev_count, int) and prev_count >= 1:
                check_count = prev_count + 1

        if previous_sha is None:
            original = previous_metadata.get("sha256")
            if isinstance(original, str):
                previous_sha = original
        if baseline_sha is None:
            baseline_candidate = previous_metadata.get("sha256")
            if isinstance(baseline_candidate, str):
                baseline_sha = baseline_candidate

        if previous_sha is not None:
            matches_previous = previous_sha == current_sha
            if check_count == 1:
                check_count = 2

    if baseline_sha is None:
        baseline_sha = current_sha

    record = {
        "baseline_sha256": baseline_sha,
        "previous_sha256": previous_sha,
        "current_sha256": current_sha,
        "matches_previous": matches_previous,
        "checked_at": now,
        "check_count": check_count,
    }
    return record


def write_meta_yaml(
    output_path: Path,
    *,
    command: str,
    config: Mapping[str, Any],
    row_count: int,
    column_count: int,
    meta_path: Path | None = None,
    status: Literal["success", "error"] = "success",
    error: str | None = None,
    include_hash: bool = True,
    warnings: Sequence[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Writes dataset metadata to a YAML file next to the output file.

    Args:
        output_path: The path to the generated CSV file.
        command: The command-line invocation that generated the data.
        config: A normalized configuration dictionary from the CLI arguments.
        row_count: The number of rows persisted to the CSV file.
        column_count: The number of columns in the CSV file.
        meta_path: An optional destination for the YAML file. If omitted, the
            function writes to `<output_path>.meta.yaml`.

    Returns:
        The path to the written metadata file.

    Other Parameters
    ----------------
    status:
        Execution outcome recorded in the metadata. Defaults to ``"success"``
        and should be set to ``"error"`` when the CLI aborts prematurely.
    error:
        Optional descriptive message associated with an error outcome.
    include_hash:
        When :data:`True`, compute and persist the SHA-256 digest alongside a
        determinism record. Disable when the output file was not produced.
    warnings:
        Optional collection of warning messages emitted during execution. The
        sequence is preserved as provided and intended for non-fatal issues that
        should be visible to downstream consumers of the metadata.
    extra:
        Additional key-value pairs appended to the metadata record. This is
        useful for storing progress checkpoints that allow long-running jobs to
        resume without starting from scratch.
    """

    default_meta_path = output_path.with_name(f"{output_path.name}.meta.yaml")
    target = meta_path or default_meta_path
    previous_metadata = _load_previous_metadata(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "config": dict(config),
        "output": str(output_path),
        "rows": row_count,
        "columns": column_count,
        "status": status,
    }

    if error:
        metadata["error"] = error

    if warnings:
        metadata["warnings"] = list(warnings)

    if extra:
        for key, value in extra.items():
            metadata[key] = value

    if include_hash:
        metadata["sha256"] = _file_sha256(output_path)
        metadata["determinism"] = _determinism_record(
            current_sha=str(metadata["sha256"]),
            previous_metadata=previous_metadata,
        )
    else:
        metadata["sha256"] = None

    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, allow_unicode=True, sort_keys=False)

    LOGGER.info("Metadata written to %s", target)
    return target
