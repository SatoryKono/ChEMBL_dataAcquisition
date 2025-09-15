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
        sep=",",
        encoding="utf-8",
    )

    assert output_path.exists()
