"""Tests for the pubmed_main CLI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence, List

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.pubmed_main as pm  # type: ignore  # noqa: E402
from library.pubmed_client import PubMedRecord  # type: ignore  # noqa: E402


def test_run_creates_output_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run() should create missing parent directories for the output file."""

    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"PMID": ["1"]}).to_csv(input_csv, index=False)

    def fake_fetch_pubmed_records(
        pmids: Sequence[str], *, client: Any, batch_size: int
    ) -> List[PubMedRecord]:
        return [
            PubMedRecord(
                pmid="1",
                doi=None,
                title=None,
                abstract=None,
                journal=None,
                publication_types=["Journal Article"],
            )
        ]

    def fake_fetch_semantic_scholar_records(
        pmids: Sequence[str], *, client: Any
    ) -> List[Any]:
        return []

    def fake_fetch_openalex_records(
        pmids: Sequence[str], *, client: Any
    ) -> List[Any]:
        return []

    def fake_fetch_crossref_records(
        dois: Sequence[str], *, client: Any
    ) -> List[Any]:
        return []

    monkeypatch.setattr(pm, "fetch_pubmed_records", fake_fetch_pubmed_records)
    monkeypatch.setattr(pm, "fetch_semantic_scholar_records", fake_fetch_semantic_scholar_records)
    monkeypatch.setattr(pm, "fetch_openalex_records", fake_fetch_openalex_records)
    monkeypatch.setattr(pm, "fetch_crossref_records", fake_fetch_crossref_records)

    output_path = tmp_path / "out" / "result.csv"

    pm.run(
        input_path=input_csv,
        output_path=output_path,
        column="PMID",
        batch_size=1,
        sleep=0.0,
        openalex_rps=0.0,
        crossref_rps=0.0,
        from_chembl=False,
        chembl_chunk_size=5,
        sep=",",
        encoding="utf-8",
    )

    assert output_path.exists()


def test_run_with_chembl_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run() should merge ChEMBL document metadata when ``from_chembl`` is set."""

    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"chembl_id": ["DOC1"]}).to_csv(input_csv, index=False)

    def fake_get_documents(ids: Sequence[str], *, cfg: Any, client: Any, chunk_size: int, timeout: Any | None = None) -> pd.DataFrame:  # noqa: D401
        return pd.DataFrame(
            {
                "document_chembl_id": ["DOC1"],
                "title": ["T"],
                "abstract": ["A"],
                "doi": ["10.1/doi1"],
                "year": [2020],
                "journal": ["J"],
                "journal_abbrev": ["J"],
                "volume": ["1"],
                "issue": ["1"],
                "first_page": ["1"],
                "last_page": ["2"],
                "pubmed_id": ["1"],
                "authors": ["A"],
                "source": ["ChEMBL"],
            }
        )

    def fake_fetch_pubmed_records(pmids: Sequence[str], *, client: Any, batch_size: int) -> List[PubMedRecord]:
        return [
            PubMedRecord(
                pmid="1",
                doi=None,
                title=None,
                abstract=None,
                journal=None,
                publication_types=["Journal Article"],
            )
        ]

    def fake_fetch_semantic_scholar_records(pmids: Sequence[str], *, client: Any) -> List[Any]:
        return []

    def fake_fetch_openalex_records(pmids: Sequence[str], *, client: Any) -> List[Any]:
        return []

    def fake_fetch_crossref_records(dois: Sequence[str], *, client: Any) -> List[Any]:
        return []

    monkeypatch.setattr(pm, "get_documents", fake_get_documents)
    monkeypatch.setattr(pm, "fetch_pubmed_records", fake_fetch_pubmed_records)
    monkeypatch.setattr(pm, "fetch_semantic_scholar_records", fake_fetch_semantic_scholar_records)
    monkeypatch.setattr(pm, "fetch_openalex_records", fake_fetch_openalex_records)
    monkeypatch.setattr(pm, "fetch_crossref_records", fake_fetch_crossref_records)

    output_path = tmp_path / "out.csv"
    pm.run(
        input_path=input_csv,
        output_path=output_path,
        column="chembl_id",
        batch_size=1,
        sleep=0.0,
        openalex_rps=0.0,
        crossref_rps=0.0,
        from_chembl=True,
        chembl_chunk_size=5,
        sep=",",
        encoding="utf-8",
    )
    df = pd.read_csv(output_path)
    assert "ChEMBL.document_chembl_id" in df.columns
