"""High level helpers for retrieving ChEMBL metadata."""

from __future__ import annotations

import logging
from typing import Callable, Iterable, Iterator, List

import pandas as pd

try:  # pragma: no cover - allow flat imports during testing
    from .chembl_client import ChemblClient
except ImportError:  # pragma: no cover
    from chembl_client import ChemblClient  # type: ignore[no-redef]

LOGGER = logging.getLogger(__name__)


def _chunked(values: Iterable[str], chunk_size: int) -> Iterator[List[str]]:
    """Yield ``values`` in lists of at most ``chunk_size`` elements."""

    chunk: List[str] = []
    for value in values:
        chunk.append(value)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _fetch_dataframe(
    *,
    fetch: Callable[[Iterable[str]], List[dict[str, object]]],
    identifiers: Iterable[str],
    chunk_size: int,
    log_label: str,
    dedupe_column: str,
) -> pd.DataFrame:
    records: List[dict[str, object]] = []
    for chunk in _chunked(identifiers, chunk_size):
        LOGGER.info("Fetching %d %s from ChEMBL", len(chunk), log_label)
        records.extend(fetch(chunk))

    if not records:
        LOGGER.warning("No %s were retrieved from the API", log_label)
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if dedupe_column in df.columns:
        df = df.drop_duplicates(subset=[dedupe_column]).sort_values(dedupe_column)
    df.reset_index(drop=True, inplace=True)
    return df


def get_assays(
    client: ChemblClient,
    assay_ids: Iterable[str],
    *,
    chunk_size: int = 20,
) -> pd.DataFrame:
    """Fetch assay metadata for ``assay_ids``.

    Parameters
    ----------
    client:
        Instance of :class:`ChemblClient` responsible for network requests.
    assay_ids:
        Iterable of assay identifiers to fetch from the ChEMBL API.
    chunk_size:
        Number of identifiers to process in a single batch.  This controls the
        log granularity and can be tuned for improved determinism when dealing
        with flaky network conditions.
    """

    return _fetch_dataframe(
        fetch=client.fetch_many,
        identifiers=assay_ids,
        chunk_size=chunk_size,
        log_label="assays",
        dedupe_column="assay_chembl_id",
    )


def get_activities(
    client: ChemblClient,
    activity_ids: Iterable[str],
    *,
    chunk_size: int = 20,
) -> pd.DataFrame:
    """Fetch activity metadata for ``activity_ids``.

    Parameters
    ----------
    client:
        Instance of :class:`ChemblClient` responsible for network requests.
    activity_ids:
        Iterable of activity identifiers to fetch from the ChEMBL API.
    chunk_size:
        Number of identifiers to process in a single batch.  This controls the
        log granularity and can be tuned for improved determinism when dealing
        with flaky network conditions.
    """

    return _fetch_dataframe(
        fetch=client.fetch_many_activities,
        identifiers=activity_ids,
        chunk_size=chunk_size,
        log_label="activities",
        dedupe_column="activity_chembl_id",
    )
