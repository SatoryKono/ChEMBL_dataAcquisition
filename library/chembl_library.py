"""High level helpers for retrieving ChEMBL metadata."""

from __future__ import annotations

import logging
from typing import Callable, Iterable, Iterator, List, Sequence

import pandas as pd

try:  # pragma: no cover - allow flat imports during testing
    from .chembl_client import ChemblClient  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    from chembl_client import ChemblClient  # type: ignore[import-not-found, no-redef]

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
) -> Iterator[pd.DataFrame]:
    """Yield records retrieved from ChEMBL in DataFrame chunks.

    The helper streams API responses by requesting ``chunk_size`` identifiers at
    a time and yielding each batch as a :class:`pandas.DataFrame`.  Records are
    deduplicated according to ``dedupe_column`` so that downstream consumers can
    append results directly to CSV files without introducing duplicate rows.

    Args:
        fetch: Callable performing the API request for a batch of identifiers.
        identifiers: Iterable producing ChEMBL identifiers.
        chunk_size: Maximum number of identifiers passed to ``fetch`` per
            request.
        log_label: Human readable label used in progress messages.
        dedupe_column: Column name identifying unique records in the response.

    Yields:
        DataFrames containing at most ``chunk_size`` records that are unique with
        respect to ``dedupe_column``.  Empty DataFrames are skipped.
    """

    seen: set[str] = set()
    for chunk in _chunked(identifiers, chunk_size):
        LOGGER.info("Fetching %d %s from ChEMBL", len(chunk), log_label)
        payload = fetch(chunk)
        if not payload:
            continue
        df = pd.DataFrame(payload)
        if dedupe_column in df.columns:
            df = df.drop_duplicates(subset=[dedupe_column])
            if seen:
                df = df[~df[dedupe_column].isin(seen)]
        if df.empty:
            continue
        if dedupe_column in df.columns:
            values = df[dedupe_column].dropna().map(str).tolist()
            seen.update(values)
            df = df.sort_values(dedupe_column)
        df.reset_index(drop=True, inplace=True)
        yield df


def _collect_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Materialise ``frames`` into a single DataFrame preserving column order."""

    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0].reset_index(drop=True)
    return pd.concat(frames, ignore_index=True)


def get_assays(
    client: ChemblClient,
    assay_ids: Iterable[str],
    *,
    chunk_size: int = 20,
) -> pd.DataFrame:
    """Fetches assay metadata for a given list of assay ChEMBL IDs.

    Args:
        client: An instance of the ChemblClient.
        assay_ids: An iterable of assay ChEMBL IDs.
        chunk_size: The number of IDs to process in a single batch.

    Returns:
        A pandas DataFrame containing the assay metadata.
    """

    frames = list(
        _fetch_dataframe(
            fetch=client.fetch_many,
            identifiers=assay_ids,
            chunk_size=chunk_size,
            log_label="assays",
            dedupe_column="assay_chembl_id",
        )
    )
    return _collect_frames(frames)


def get_activities(
    client: ChemblClient,
    activity_ids: Iterable[str],
    *,
    chunk_size: int = 20,
) -> pd.DataFrame:
    """Fetches activity metadata for a given list of activity ChEMBL IDs.

    Args:
        client: An instance of the ChemblClient.
        activity_ids: An iterable of activity ChEMBL IDs.
        chunk_size: The number of IDs to process in a single batch.

    Returns:
        A pandas DataFrame containing the activity metadata.
    """

    frames = list(
        _fetch_dataframe(
            fetch=client.fetch_many_activities,
            identifiers=activity_ids,
            chunk_size=chunk_size,
            log_label="activities",
            dedupe_column="activity_chembl_id",
        )
    )
    return _collect_frames(frames)


def get_testitems(
    client: ChemblClient,
    molecule_ids: Iterable[str],
    *,
    chunk_size: int = 20,
) -> pd.DataFrame:
    """Fetches molecule metadata for a given list of molecule ChEMBL IDs.

    Args:
        client: An instance of the ChemblClient.
        molecule_ids: An iterable of molecule ChEMBL IDs.
        chunk_size: The number of IDs to process in a single batch.

    Returns:
        A pandas DataFrame containing the molecule metadata.
    """

    frames = list(
        _fetch_dataframe(
            fetch=client.fetch_many_molecules,
            identifiers=molecule_ids,
            chunk_size=chunk_size,
            log_label="molecules",
            dedupe_column="molecule_chembl_id",
        )
    )
    return _collect_frames(frames)


def stream_assays(
    client: ChemblClient,
    assay_ids: Iterable[str],
    *,
    chunk_size: int = 20,
) -> Iterator[pd.DataFrame]:
    """Stream assay metadata from ChEMBL in DataFrame chunks."""

    yield from _fetch_dataframe(
        fetch=client.fetch_many,
        identifiers=assay_ids,
        chunk_size=chunk_size,
        log_label="assays",
        dedupe_column="assay_chembl_id",
    )


def stream_activities(
    client: ChemblClient,
    activity_ids: Iterable[str],
    *,
    chunk_size: int = 20,
) -> Iterator[pd.DataFrame]:
    """Stream activity metadata from ChEMBL in DataFrame chunks."""

    yield from _fetch_dataframe(
        fetch=client.fetch_many_activities,
        identifiers=activity_ids,
        chunk_size=chunk_size,
        log_label="activities",
        dedupe_column="activity_chembl_id",
    )


def stream_testitems(
    client: ChemblClient,
    molecule_ids: Iterable[str],
    *,
    chunk_size: int = 20,
) -> Iterator[pd.DataFrame]:
    """Stream molecule metadata from ChEMBL in DataFrame chunks."""

    yield from _fetch_dataframe(
        fetch=client.fetch_many_molecules,
        identifiers=molecule_ids,
        chunk_size=chunk_size,
        log_label="molecules",
        dedupe_column="molecule_chembl_id",
    )
