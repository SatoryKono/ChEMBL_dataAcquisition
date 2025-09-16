"""Tests for the pubmed_main CLI helpers."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.pubmed_main as pm  # type: ignore  # noqa: E402
from library.pubmed_client import PubMedRecord  # type: ignore  # noqa: E402
from library.semantic_scholar_client import SemanticScholarRecord  # type: ignore  # noqa: E402
from library.openalex_client import OpenAlexRecord  # type: ignore  # noqa: E402
from library.crossref_client import CrossrefRecord  # type: ignore  # noqa: E402


def _make_args(
    command: str, *, input_path: Path, output_path: Path, column: str
) -> argparse.Namespace:
    return argparse.Namespace(
        command=command,
        input=input_path,
        output=output_path,
        column=column,
        workers=1,
    )


def test_run_pubmed_creates_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_pubmed_command should create the parent directory for the output file."""

    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"PMID": ["1"]}).to_csv(input_csv, index=False)

    pubmed_record = PubMedRecord(
        pmid="1",
        doi="10.1/doi1",
        title="Title",
        abstract="Abstract",
        journal="Journal",
        journal_abbrev="J",
        volume="1",
        issue="1",
        start_page="1",
        end_page="2",
        issn="1234-5678",
        publication_types=["Journal Article"],
        mesh_descriptors=["Descriptor"],
        mesh_qualifiers=["Qualifier"],
        chemical_list=["Chem"],
        year_completed="2020",
        month_completed="01",
        day_completed="01",
        year_revised=None,
        month_revised=None,
        day_revised=None,
        error=None,
    )
    scholar_record = SemanticScholarRecord(
        pmid="1",
        doi="10.1/doi1",
        publication_types=["Journal Article"],
        venue="Venue",
        paper_id="S1",
        external_ids={"PMID": "1", "DOI": "10.1/doi1"},
        error=None,
    )
    openalex_record = OpenAlexRecord(
        pmid="1",
        doi="10.1/doi1",
        publication_types=["journal-article"],
        type_crossref="journal-article",
        genre="journal-article",
        venue="Venue",
        mesh_descriptors=[],
        mesh_qualifiers=[],
        work_id="W1",
        error=None,
    )
    crossref_record = CrossrefRecord(
        doi="10.1/doi1",
        type="journal-article",
        subtype=None,
        title="Title",
        subtitle=None,
        subject=["Biology"],
        error=None,
    )

    def fake_gather(
        pmids: Sequence[str], *, cfg: dict[str, Any]
    ) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
        return [pubmed_record], [scholar_record], [openalex_record], [crossref_record]

    monkeypatch.setattr(pm, "_gather_pubmed_sources", fake_gather)

    output_path = tmp_path / "out" / "result.csv"
    args = _make_args(
        "pubmed", input_path=input_csv, output_path=output_path, column="PMID"
    )
    config = deepcopy(pm.DEFAULT_CONFIG)

    pm.run_pubmed_command(args, config)

    assert output_path.exists()


def test_run_all_merges_chembl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run_all_command should merge ChEMBL data into the output."""

    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"chembl_id": ["DOC1"]}).to_csv(input_csv, index=False)

    chembl_df = pd.DataFrame(
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
            "authors": ["Authors"],
            "source": ["ChEMBL"],
        }
    )

    pubmed_record = PubMedRecord(
        pmid="1",
        doi="10.1/doi1",
        title="Title",
        abstract="Abstract",
        journal="Journal",
        journal_abbrev="J",
        volume="1",
        issue="1",
        start_page="1",
        end_page="2",
        issn="1234-5678",
        publication_types=["Journal Article"],
        mesh_descriptors=[],
        mesh_qualifiers=[],
        chemical_list=[],
        year_completed=None,
        month_completed=None,
        day_completed=None,
        year_revised=None,
        month_revised=None,
        day_revised=None,
        error=None,
    )

    def fake_get_documents(ids: Sequence[str], *, cfg: Any, client: Any, chunk_size: int, timeout: float) -> pd.DataFrame:  # type: ignore[override]
        return chembl_df

    def fake_gather(
        pmids: Sequence[str], *, cfg: dict[str, Any]
    ) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
        return [pubmed_record], [], [], []

    monkeypatch.setattr(pm, "get_documents", fake_get_documents)
    monkeypatch.setattr(pm, "_gather_pubmed_sources", fake_gather)

    output_path = tmp_path / "output.csv"
    args = _make_args(
        "all", input_path=input_csv, output_path=output_path, column="chembl_id"
    )
    args.workers = 1
    config = deepcopy(pm.DEFAULT_CONFIG)

    pm.run_all_command(args, config)

    df = pd.read_csv(output_path)
    assert "ChEMBL.document_chembl_id" in df.columns
    assert df.loc[0, "ChEMBL.document_chembl_id"] == "DOC1"


def test_global_cli_overrides_before_command() -> None:
    """Global CLI flags provided before the command should update the config."""

    parser = pm.build_parser()
    args = parser.parse_args(["--batch-size", "25", "all"])
    config = deepcopy(pm.DEFAULT_CONFIG)

    pm.apply_cli_overrides(args, config)

    assert config["pubmed"]["batch_size"] == 25


def test_chembl_global_cli_overrides() -> None:
    """Global ChEMBL flags should work regardless of argument order."""

    parser = pm.build_parser()
    args = parser.parse_args(["--chunk-size", "8", "chembl"])
    config = deepcopy(pm.DEFAULT_CONFIG)

    pm.apply_cli_overrides(args, config)

    assert config["chembl"]["chunk_size"] == 8
