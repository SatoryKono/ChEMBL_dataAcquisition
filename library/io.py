"""Streaming CSV helpers for ChEMBL data acquisition.

The :func:`read_ids` generator reads identifiers lazily from disk, ensuring
that very large input files do not need to be loaded into memory all at once.
The function mirrors the behaviour of :func:`library.io_utils.read_ids` but
avoids materialising the full list of identifiers.  Values are normalised to
upper case and deduplicated while preserving their first-seen order.

Algorithm Notes
---------------
1. Stream the CSV file using :class:`csv.DictReader` with the configured
   delimiter and encoding.
2. Normalise each value to upper case and strip surrounding whitespace.
3. Yield unique identifiers, skipping empty entries.
4. Stop after ``limit`` unique identifiers when requested.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from collections.abc import Callable
from typing import Iterator, Set

from .io_utils import CsvConfig

LOGGER = logging.getLogger(__name__)


def read_ids(
    path: Path,
    column: str,
    cfg: CsvConfig,
    *,
    limit: int | None = None,
    normalise: Callable[[str], str] | None = str.upper,
) -> Iterator[str]:
    """Yield unique identifiers from ``column`` of ``path`` lazily.

    Parameters
    ----------
    path:
        Path to the CSV file containing identifiers.
    column:
        Name of the column holding the identifiers of interest.
    cfg:
        CSV configuration specifying delimiter and encoding.
    limit:
        Optional maximum number of **unique** identifiers to yield. ``None``
        disables limiting and streams the entire file.
    normalise:
        Callable applied to each non-empty cell before deduplication and
        yielding. The default converts values to upper case. Pass ``None`` to
        preserve the original casing while still trimming whitespace.

    Yields
    ------
    str
        Upper-case identifier values with surrounding whitespace removed.

    Raises
    ------
    KeyError
        If ``column`` is missing from the CSV header.
    """

    LOGGER.debug("Reading identifiers from %s (column=%s)", path, column)
    with path.open("r", encoding=cfg.encoding, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=cfg.sep)
        if reader.fieldnames is None or column not in reader.fieldnames:
            msg = f"Missing required column '{column}'"
            raise KeyError(msg)

        seen: Set[str] = set()
        yielded = 0
        for row in reader:
            raw_value = (row.get(column) or "").strip()
            if not raw_value:
                continue
            normalised = normalise(raw_value) if normalise is not None else raw_value
            if normalised in seen:
                continue
            seen.add(normalised)
            yield normalised
            yielded += 1
            if limit is not None and yielded >= limit:
                break
