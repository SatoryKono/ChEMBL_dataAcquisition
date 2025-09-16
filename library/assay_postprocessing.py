"""Post-processing helpers for normalised assay tables."""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

LOGGER = logging.getLogger(__name__)


REQUIRED_COLUMNS: List[str] = [
    "assay_chembl_id",
    "document_chembl_id",
    "target_chembl_id",
]


def postprocess_assays(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` augmented with ``assay_with_same_target``.

    The function groups rows by ``document_chembl_id`` and ``target_chembl_id``
    and counts how many assays share the same pair.  Missing identifiers are
    treated as their own group so that orphaned records still receive a count.
    """

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        msg = f"Missing required columns for post-processing: {', '.join(missing)}"
        raise KeyError(msg)

    if df.empty:
        LOGGER.info("Received empty DataFrame for post-processing")
        result = df.copy()
        result["assay_with_same_target"] = pd.Series(dtype="int64")
        return result

    result = df.copy()

    def _strip(value: object) -> object:
        if isinstance(value, str):
            text = value.strip()
            return text if text else None
        return value

    result["__doc_norm"] = result["document_chembl_id"].map(_strip)
    result["__target_norm"] = result["target_chembl_id"].map(_strip)
    counts = (
        result.groupby(["__doc_norm", "__target_norm"], dropna=False)["assay_chembl_id"]
        .transform("count")
        .astype("int64")
    )
    result["assay_with_same_target"] = counts
    result.drop(columns=["__doc_norm", "__target_norm"], inplace=True)
    return result
