"""Normalisation helpers for GtoPdb API responses.

Each function accepts raw JSON structures and converts them into deterministic
:class:`pandas.DataFrame` instances ready for serialisation.  Lists are sorted by
stable keys to ensure repeated runs produce byte-identical outputs.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# Targets and related resources


def normalise_targets(items: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Return a DataFrame with a single row per target.

    Parameters
    ----------
    items:
        Iterable of target dictionaries as returned by ``/targets``.
    """

    columns = [
        "targetId",
        "name",
        "targetType",
        "family",
        "species",
        "description",
    ]
    records = [{col: t.get(col) for col in columns} for t in items]
    df = pd.DataFrame(records, columns=columns)
    if not df.empty:
        df = df.sort_values("targetId").reset_index(drop=True)
    return df


def normalise_synonyms(target_id: int, items: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    columns = ["targetId", "synonym", "source"]
    records = []
    for item in items or []:
        records.append(
            {
                "targetId": target_id,
                "synonym": item.get("synonym"),
                "source": item.get("synonymType") or item.get("source"),
            }
        )
    df = pd.DataFrame(records, columns=columns)
    if not df.empty:
        df = df.sort_values(["targetId", "synonym"]).reset_index(drop=True)
    return df


def normalise_interactions(
    target_id: int, items: Iterable[Dict[str, Any]]
) -> pd.DataFrame:
    columns = [
        "targetId",
        "ligandId",
        "type",
        "action",
        "affinity",
        "affinityParameter",
        "species",
        "ligandType",
        "approved",
        "primaryTarget",
    ]
    records = []
    for item in items or []:
        records.append(
            {
                "targetId": target_id,
                "ligandId": item.get("ligandId"),
                "type": item.get("type"),
                "action": item.get("action"),
                "affinity": item.get("affinity"),
                "affinityParameter": item.get("affinityParameter"),
                "species": item.get("species"),
                "ligandType": item.get("ligandType"),
                "approved": item.get("approved"),
                "primaryTarget": item.get("primaryTarget"),
            }
        )
    df = pd.DataFrame(records, columns=columns)
    if not df.empty:
        df = df.sort_values(["targetId", "ligandId"]).reset_index(drop=True)
    return df


__all__ = [
    "normalise_targets",
    "normalise_synonyms",
    "normalise_interactions",
]
