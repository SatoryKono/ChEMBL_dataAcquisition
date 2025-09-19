"""Utilities for verifying deterministic CSV output and metadata."""

from __future__ import annotations

import argparse
import hashlib
import logging
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import yaml

if __package__ in {None, ""}:
    from _path_utils import ensure_project_root as _ensure_project_root

    _ensure_project_root()

from library.io_utils import CsvConfig, write_rows  # noqa: E402
from library.logging_utils import configure_logging  # noqa: E402
from library.metadata import write_meta_yaml  # noqa: E402

LOGGER = logging.getLogger(__name__)


def _hash_file(path: Path) -> str:
    """Return the SHA-256 hash of ``path``."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_rows() -> Sequence[Dict[str, Any]]:
    """Provide a deterministic set of rows for testing."""

    return [
        {"id": 1, "names": ["alpha", "beta"]},
        {"id": 2, "names": ["gamma", "delta"]},
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for the determinism checker.

    Args:
        argv: A sequence of command-line arguments. If None, `sys.argv` is used.

    Returns:
        An `argparse.Namespace` object containing the parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("_determinism_check.csv"),
        help="Destination CSV used for the comparison",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Preserve the generated CSV and metadata for inspection",
    )
    return parser.parse_args(argv)


def _write_and_record(
    output_path: Path,
    rows: Sequence[Dict[str, Any]],
    columns: Sequence[str],
    cfg: CsvConfig,
    *,
    command: str,
) -> str:
    """Write ``rows`` to ``output_path`` and persist metadata."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_rows(output_path, rows, columns, cfg)
    digest = _hash_file(output_path)
    LOGGER.info("Wrote %s with hash %s", output_path, digest)
    write_meta_yaml(
        output_path,
        command=command,
        config={"columns": list(columns), "list_format": cfg.list_format},
        row_count=len(rows),
        column_count=len(columns),
    )
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the determinism check and updates the `.meta.yaml` file with the outcome.

    Args:
        argv: A sequence of command-line arguments. If None, `sys.argv` is used.

    Returns:
        An exit code, 0 for success and 1 for failure.
    """

    args = parse_args(argv)
    configure_logging("INFO")

    cfg = CsvConfig(sep=",", encoding="utf-8", list_format="json")
    rows = _sample_rows()
    columns = ("id", "names")
    output_path = args.output.resolve()
    meta_path = output_path.with_suffix(f"{output_path.suffix}.meta.yaml")
    command = " ".join(
        shlex.quote(part) for part in (sys.argv[0], *(argv or sys.argv[1:]))
    )

    hashes: list[str] = []
    try:
        hashes.append(
            _write_and_record(output_path, rows, columns, cfg, command=command)
        )
        hashes.append(
            _write_and_record(output_path, rows, columns, cfg, command=command)
        )

        if hashes[0] != hashes[1]:
            msg = "Non-deterministic output detected: %s != %s" % (hashes[0], hashes[1])
            raise RuntimeError(msg)

        metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        determinism = metadata.get("determinism", {})
        matches_previous = determinism.get("matches_previous")
        if matches_previous is False:
            raise RuntimeError("Metadata reports non-deterministic output")
        LOGGER.info("Determinism check succeeded: %s", determinism)
    finally:
        if not args.keep_artifacts:
            output_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
