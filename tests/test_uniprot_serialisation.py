"""Ensure UniProt exports rely on native Python collections for serialisation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from library.io_utils import CsvConfig, write_rows


@pytest.mark.parametrize("list_format", ["json", "pipe"])
def test_uniprot_serialisation_formats(tmp_path: Path, list_format: str) -> None:
    """Verify serialisation of isoform and ortholog payloads across formats."""

    cfg = CsvConfig(sep=",", encoding="utf-8", list_format=list_format)
    ortholog_entry = {
        "dn": 0.1,
        "ds": None,
        "homology_type": "ortholog",
        "is_high_confidence": False,
        "perc_id": 55.5,
        "perc_pos": 60.0,
        "source_db": "Ensembl",
        "target_ensembl_gene_id": "ENSR1",
        "target_gene_symbol": "GeneR",
        "target_species": "rat",
        "target_uniprot_id": "Q1",
    }
    rows = [
        {
            "parent_uniprot_id": "P12345",
            "isoform_uniprot_id": "P12345-1",
            "isoform_name": "Isoform 1",
            "isoform_synonyms": ["Alpha", "Beta"],
            "is_canonical": True,
            "orthologs_json": [ortholog_entry],
            "orthologs_count": 1,
        }
    ]
    columns: list[str] = [
        "parent_uniprot_id",
        "isoform_uniprot_id",
        "isoform_name",
        "isoform_synonyms",
        "is_canonical",
        "orthologs_json",
        "orthologs_count",
    ]

    output_path = tmp_path / f"serialised_{list_format}.csv"
    write_rows(output_path, rows, columns, cfg)

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=cfg.sep)
        result = next(reader)

    if list_format == "json":
        assert json.loads(result["isoform_synonyms"]) == ["Alpha", "Beta"]
        ortholog_payload = json.loads(result["orthologs_json"])
        assert isinstance(ortholog_payload, list)
        assert ortholog_payload[0]["is_high_confidence"] is False
        assert ortholog_payload[0]["perc_id"] == pytest.approx(55.5)
    else:
        assert result["isoform_synonyms"] == "Alpha|Beta"
        fragments = result["orthologs_json"].split("|")
        parsed = [json.loads(fragment.replace("\\|", "|")) for fragment in fragments]
        assert parsed[0]["target_species"] == "rat"
        assert parsed[0]["perc_id"] == pytest.approx(55.5)

    assert result["is_canonical"] == "True"
    assert result["orthologs_count"] == "1"
