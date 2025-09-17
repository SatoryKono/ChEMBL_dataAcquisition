"""Utilities for merging document metadata from multiple bibliographic sources."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

import pandas as pd

from .crossref_client import CrossrefRecord
from .openalex_client import OpenAlexRecord
from .pubmed_client import PubMedRecord, score_publication_types
from .semantic_scholar_client import SemanticScholarRecord

LOGGER = logging.getLogger(__name__)

DOCUMENT_SCHEMA_COLUMNS: List[str] = [
    "PubMed.PMID",
    "PubMed.DOI",
    "PubMed.ArticleTitle",
    "PubMed.Abstract",
    "PubMed.JournalTitle",
    "PubMed.JournalISOAbbrev",
    "PubMed.Volume",
    "PubMed.Issue",
    "PubMed.StartPage",
    "PubMed.EndPage",
    "PubMed.ISSN",
    "PubMed.PublicationType",
    "PubMed.MeSH_Descriptors",
    "PubMed.MeSH_Qualifiers",
    "PubMed.ChemicalList",
    "PubMed.YearCompleted",
    "PubMed.MonthCompleted",
    "PubMed.DayCompleted",
    "PubMed.YearRevised",
    "PubMed.MonthRevised",
    "PubMed.DayRevised",
    "PubMed.Error",
    "scholar.PMID",
    "scholar.DOI",
    "scholar.PublicationTypes",
    "scholar.Venue",
    "scholar.SemanticScholarId",
    "scholar.ExternalIds",
    "scholar.Error",
    "OpenAlex.PMID",
    "OpenAlex.DOI",
    "OpenAlex.PublicationTypes",
    "OpenAlex.TypeCrossref",
    "OpenAlex.Genre",
    "OpenAlex.Venue",
    "OpenAlex.MeshDescriptors",
    "OpenAlex.MeshQualifiers",
    "OpenAlex.Id",
    "OpenAlex.Error",
    "crossref.DOI",
    "crossref.Type",
    "crossref.Subtype",
    "crossref.Title",
    "crossref.Subtitle",
    "crossref.Subject",
    "crossref.Error",
    "publication_types_normalised",
    "publication_review_score",
    "publication_experimental_score",
    "publication_class",
]

CH_EMBL_COLUMNS: List[str] = [
    "ChEMBL.document_chembl_id",
    "ChEMBL.title",
    "ChEMBL.abstract",
    "ChEMBL.doi",
    "ChEMBL.year",
    "ChEMBL.journal",
    "ChEMBL.journal_abbrev",
    "ChEMBL.volume",
    "ChEMBL.issue",
    "ChEMBL.first_page",
    "ChEMBL.last_page",
    "ChEMBL.pubmed_id",
    "ChEMBL.authors",
    "ChEMBL.source",
]


SEMANTIC_SCHOLAR_COLUMNS: List[str] = [
    "scholar.PMID",
    "scholar.DOI",
    "scholar.PublicationTypes",
    "scholar.Venue",
    "scholar.SemanticScholarId",
    "scholar.ExternalIds",
    "scholar.Error",
]


def _normalise_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = doi.strip().lower()
    return doi or None


def merge_metadata(
    pubmed_records: Sequence[PubMedRecord],
    scholar_records: Sequence[SemanticScholarRecord],
    openalex_records: Sequence[OpenAlexRecord],
    crossref_records: Sequence[CrossrefRecord],
    *,
    max_workers: int = 1,
    progress_callback: Callable[[int], None] | None = None,
) -> List[Dict[str, Any]]:
    """Merge per-source metadata into row dictionaries suitable for CSV output.

    Parameters
    ----------
    pubmed_records:
        PubMed records to merge with partner data sources.
    scholar_records:
        Records returned from the Semantic Scholar API.
    openalex_records:
        Records returned from the OpenAlex API.
    crossref_records:
        Records returned from the Crossref API.
    max_workers:
        Maximum number of worker threads used when building merged rows.
    progress_callback:
        Optional callable invoked with the number of processed records after
        each merge step. Designed for progress bar integrations.
    """

    scholar_map = {rec.pmid: rec for rec in scholar_records}
    openalex_map = {rec.pmid: rec for rec in openalex_records}
    crossref_map: Dict[str, CrossrefRecord] = {}
    for crossref_record in crossref_records:
        if crossref_record.doi:
            key = _normalise_doi(crossref_record.doi)
            if key:
                crossref_map[key] = crossref_record

    def _build_row(record: PubMedRecord) -> Dict[str, Any]:
        row = record.to_dict()
        review_score_total = 0
        experimental_score_total = 0
        normalised_types: List[str] = []

        review_score, experimental_score, normalised = score_publication_types(
            record.publication_types, source="pubmed"
        )
        review_score_total += review_score
        experimental_score_total += experimental_score
        normalised_types.extend(normalised)

        scholar = scholar_map.get(record.pmid)
        if scholar:
            row.update(scholar.to_dict())
            review_score, experimental_score, normalised = score_publication_types(
                scholar.publication_types, source="scholar"
            )
            review_score_total += review_score
            experimental_score_total += experimental_score
            normalised_types.extend(normalised)
        else:
            row.setdefault("scholar.PMID", record.pmid)

        openalex = openalex_map.get(record.pmid)
        if openalex:
            row.update(openalex.to_dict())
            review_score, experimental_score, normalised = score_publication_types(
                openalex.publication_types, source="openalex"
            )
            review_score_total += review_score
            experimental_score_total += experimental_score
            normalised_types.extend(normalised)
        else:
            row.setdefault("OpenAlex.PMID", record.pmid)

        doi_candidates = [
            row.get("PubMed.DOI"),
            row.get("scholar.DOI"),
            row.get("OpenAlex.DOI"),
        ]
        doi: str | None = None
        for candidate in doi_candidates:
            if isinstance(candidate, str) and candidate.strip():
                doi = candidate.strip()
                break
        if doi:
            lookup = _normalise_doi(doi)
            if lookup:
                crossref = crossref_map.get(lookup)
                if crossref:
                    row.update(crossref.to_dict())
                else:
                    row.setdefault("crossref.DOI", doi)
        row.setdefault("crossref.DOI", doi)

        deduped_types = sorted({t for t in normalised_types if t})
        row["publication_types_normalised"] = "|".join(deduped_types)
        row["publication_review_score"] = review_score_total
        row["publication_experimental_score"] = experimental_score_total
        if review_score_total and review_score_total >= experimental_score_total:
            row["publication_class"] = "review"
        elif experimental_score_total:
            row["publication_class"] = "experimental"
        else:
            row["publication_class"] = "unknown"

        return row

    if max_workers > 1 and len(pubmed_records) > 1:
        from concurrent.futures import ThreadPoolExecutor

        rows: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for row in executor.map(_build_row, pubmed_records):
                rows.append(row)
                if progress_callback is not None:
                    progress_callback(1)
    else:
        rows = []
        for pubmed_record in pubmed_records:
            rows.append(_build_row(pubmed_record))
            if progress_callback is not None:
                progress_callback(1)

    return rows


@dataclass
class DocumentsSchema:
    """Minimal validator for the merged documents table."""

    required_columns: Sequence[str]
    optional_prefixes: Sequence[str] = ("ChEMBL.",)

    def validate(self, df: pd.DataFrame) -> List[str]:
        """Return a list of validation error messages."""

        errors: List[str] = []
        for column in self.required_columns:
            if column not in df.columns:
                errors.append(f"Missing required column '{column}'")
        return errors


def build_dataframe(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Create a deterministic DataFrame from ``rows`` respecting schema order."""

    if not rows:
        df = pd.DataFrame(columns=DOCUMENT_SCHEMA_COLUMNS)
    else:
        df = pd.DataFrame(rows)
    for column in DOCUMENT_SCHEMA_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    other_columns = sorted(
        col for col in df.columns if col not in DOCUMENT_SCHEMA_COLUMNS
    )
    ordered_columns = list(DOCUMENT_SCHEMA_COLUMNS) + other_columns
    df = df.reindex(columns=ordered_columns)
    return df


def merge_with_chembl(df: pd.DataFrame, chembl_df: pd.DataFrame) -> pd.DataFrame:
    """Merge the metadata table with ChEMBL records on ``PubMed.PMID``."""

    if chembl_df.empty:
        return df

    normalised = chembl_df.rename(
        columns={
            "document_chembl_id": "ChEMBL.document_chembl_id",
            "title": "ChEMBL.title",
            "abstract": "ChEMBL.abstract",
            "doi": "ChEMBL.doi",
            "year": "ChEMBL.year",
            "journal": "ChEMBL.journal",
            "journal_abbrev": "ChEMBL.journal_abbrev",
            "volume": "ChEMBL.volume",
            "issue": "ChEMBL.issue",
            "first_page": "ChEMBL.first_page",
            "last_page": "ChEMBL.last_page",
            "pubmed_id": "PubMed.PMID",
            "authors": "ChEMBL.authors",
            "source": "ChEMBL.source",
        }
    )
    for column in CH_EMBL_COLUMNS:
        if column not in normalised.columns:
            normalised[column] = pd.NA
    if "PubMed.PMID" in normalised.columns:
        normalised["PubMed.PMID"] = normalised["PubMed.PMID"].astype("string")
    merged = normalised.merge(df, on="PubMed.PMID", how="left")
    chembl_cols = [col for col in CH_EMBL_COLUMNS if col in merged.columns]
    metadata_cols = [col for col in df.columns if col in merged.columns]
    other_cols = [
        col
        for col in merged.columns
        if col not in chembl_cols and col not in metadata_cols
    ]
    ordered = metadata_cols + other_cols + chembl_cols
    merged = merged.reindex(columns=ordered)
    return merged


def dataframe_to_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all DataFrame columns to strings for deterministic CSV export."""

    result = df.copy()
    for column in result.columns:
        result[column] = result[column].astype("string")
    return result


def compute_file_hash(path: Path) -> str:
    """Return the SHA256 hash of ``path`` for provenance tracking."""

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Return basic quality metrics for ``df``."""

    total = len(df)
    doi_columns = [
        column
        for column in (
            "PubMed.DOI",
            "scholar.DOI",
            "OpenAlex.DOI",
            "crossref.DOI",
            "ChEMBL.doi",
        )
        if column in df.columns
    ]
    if doi_columns:
        doi_present = pd.Series(False, index=df.index, dtype=bool)
        for column in doi_columns:
            values = df[column].fillna("").astype(str).str.strip()
            doi_present = doi_present | values.ne("")
        doi_missing = int((~doi_present).sum())
    else:
        doi_missing = total
    class_counts = df.get("publication_class", pd.Series(dtype="string")).value_counts(
        dropna=False
    )
    error_columns = [col for col in df.columns if col.endswith(".Error")]
    error_summary = {
        col: int((df[col].fillna("").astype(str).str.len() > 0).sum())
        for col in error_columns
    }
    return {
        "row_count": total,
        "missing_doi": doi_missing,
        "publication_class_distribution": class_counts.to_dict(),
        "error_counts": error_summary,
    }


def save_quality_report(path: Path, report: Mapping[str, Any]) -> None:
    """Serialise ``report`` as JSON next to the main CSV file."""

    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "DOCUMENT_SCHEMA_COLUMNS",
    "CH_EMBL_COLUMNS",
    "merge_metadata",
    "build_dataframe",
    "merge_with_chembl",
    "dataframe_to_strings",
    "DocumentsSchema",
    "quality_report",
    "compute_file_hash",
    "save_quality_report",
    "SEMANTIC_SCHOLAR_COLUMNS",
]
