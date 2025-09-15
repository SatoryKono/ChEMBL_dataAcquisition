"""Utilities for verifying deterministic outputs.

This script writes a small CSV file twice using :func:`write_rows` and
confirms that the resulting hashes are identical.  It exits with a non-zero
status if the hashes differ, making it suitable for use in automated tests.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from library.io_utils import CsvConfig, write_rows  # noqa: E402

LOGGER = logging.getLogger(__name__)


def _hash_file(path: Path) -> str:
    """Return the SHA-256 hash of ``path``.

    Parameters
    ----------
    path:
        File whose contents will be hashed.
    """

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_rows() -> Sequence[Dict[str, Any]]:
    """Provide a deterministic set of rows for testing."""

    return [
        {"id": 1, "names": ["alpha", "beta"]},
        {"id": 2, "names": ["gamma", "delta"]},
    ]


def main() -> None:
    """Run the determinism check.

    The function writes the same rows twice to different temporary files and
    compares their SHA-256 hashes.  A mismatch results in a ``RuntimeError``.
    """

    logging.basicConfig(level=logging.INFO)

    cfg = CsvConfig(sep=",", encoding="utf-8", list_format="json")
    rows = _sample_rows()
    columns = ["id", "names"]

    tmp1 = Path("_det_check_1.csv")
    tmp2 = Path("_det_check_2.csv")

    write_rows(tmp1, rows, columns, cfg)
    write_rows(tmp2, rows, columns, cfg)

    hash1 = _hash_file(tmp1)
    hash2 = _hash_file(tmp2)

    LOGGER.info("hash1=%s hash2=%s", hash1, hash2)

    try:
        if hash1 != hash2:
            msg = "Non-deterministic output detected"
            raise RuntimeError(msg)
    finally:
        tmp1.unlink(missing_ok=True)
        tmp2.unlink(missing_ok=True)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
