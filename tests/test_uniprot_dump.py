from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd
import pytest
import requests_mock
import yaml

from library.orthologs import Ortholog

from uniprot_normalize import (
    extract_isoforms,
    normalize_entry,
    output_columns,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "get_uniprot_target_data.py"


def _load_sample() -> dict:
    return json.loads((ROOT / "tests/data/uniprot_entry.json").read_text())


def test_normalize_entry_sorts_lists() -> None:
    entry = _load_sample()
    headers = [">sp|P12345-1|", ">sp|P12345-2|"]
    isoforms = extract_isoforms(entry, headers)
    data = normalize_entry(entry, include_sequence=False, isoforms=isoforms)
    assert data["protein_alternative_names"] == ["Alt A", "Alt B"]
    assert data["isoform_ids"] == ["P12345-1", "P12345-2"]
    assert data["isoform_names"] == ["Isoform 1", "Isoform 2"]
    assert data["isoform_ids_all"] == ["P12345-1", "P12345-2"]
    assert data["isoforms_count"] == 2
    iso_json = json.loads(data["isoforms_json"])
    assert iso_json[0]["isoform_uniprot_id"] == "P12345-1"
    assert data["domains_pfam"] == [("PF00001", "DomainA"), ("PF00002", "DomainB")]
    assert data["features_signal_peptide"] == ["1-10"]
    assert data["features_transmembrane"] == ["20-40"]
    assert data["ptm_modified_residue"] == ["150:Phosphoserine"]


def test_extract_isoforms_parses_synonyms() -> None:
    entry = _load_sample()
    headers = [">sp|P12345-1|", ">sp|P12345-2|"]
    isoforms = extract_isoforms(entry, headers)
    assert [iso["isoform_name"] for iso in isoforms] == ["Isoform 1", "Isoform 2"]
    assert isoforms[0]["isoform_synonyms"] == ["Alpha", "Zeta"]
    assert isoforms[1]["isoform_synonyms"] == ["Beta", "Gamma"]


def test_output_columns_include_sequence() -> None:
    cols = output_columns(True)
    assert "sequence" in cols
    assert cols.index("sequence") == 17


def test_cli_writes_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from importlib.machinery import SourceFileLoader

    module = SourceFileLoader("get_uniprot_target_data", str(SCRIPT_PATH)).load_module()

    class DummyEnsemblClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            """Create a lightweight placeholder for the Ensembl client."""

        def get_orthologs(
            self, _gene_id: str, _target_species: list[str]
        ) -> list[Ortholog]:
            """Return no orthologs so the OMA fallback is exercised."""

            return []

    class DummyOmaClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            """Provide deterministic OMA responses for the test harness."""

        def get_orthologs_by_uniprot(self, _acc: str) -> list[Ortholog]:
            """Return a minimal ortholog payload."""

            return [
                Ortholog(
                    target_species="mus_musculus",
                    target_gene_symbol="GeneX",
                    target_ensembl_gene_id="ENSMUSG0000001",
                    target_uniprot_id="Q9XYZ1",
                    homology_type="ortholog_one2one",
                    perc_id=55.0,
                    perc_pos=60.0,
                    dn=0.1,
                    ds=0.2,
                    is_high_confidence=True,
                    source_db="OMA",
                )
            ]

    monkeypatch.setattr(
        "library.orthologs.EnsemblHomologyClient", DummyEnsemblClient
    )
    monkeypatch.setattr("library.orthologs.OmaClient", DummyOmaClient)

    inp = tmp_path / "inp.csv"
    inp.write_text("uniprot_id\nP12345\np12345\nP00000\n")
    out = tmp_path / "nested" / "out.csv"

    sample = _load_sample()
    with requests_mock.Mocker() as m:
        m.get(
            "https://rest.uniprot.org/uniprotkb/P12345.json",
            json=sample,
        )
        m.get(
            "https://rest.uniprot.org/uniprotkb/P00000.json",
            status_code=404,
        )
        m.get(
            re.compile("https://rest.uniprot.org/uniprotkb/stream.*"),
            [
                {"text": ">sp|P12345-1|\n>sp|P12345-2|\n"},
                {"text": ""},
            ],
        )
        module.main(
            [
                "--input",
                str(inp),
                "--output",
                str(out),
                "--with-orthologs",
            ]
        )
    iso_out = out.with_name(f"{out.stem}_isoforms.csv")
    orth_out = out.with_name(f"{out.stem}_orthologs.csv")
    meta_path = out.with_name(f"{out.name}.meta.yaml")
    iso_meta_path = iso_out.with_name(f"{iso_out.name}.meta.yaml")
    orth_meta_path = orth_out.with_name(f"{orth_out.name}.meta.yaml")
    iso_quality_path = iso_out.with_name(f"{iso_out.stem}_quality_report_table.csv")
    orth_quality_path = orth_out.with_name(
        f"{orth_out.stem}_quality_report_table.csv"
    )
    assert out.exists()
    assert iso_out.exists()
    assert orth_out.exists()
    assert meta_path.exists()
    assert iso_meta_path.exists()
    assert orth_meta_path.exists()
    assert iso_quality_path.exists()
    assert orth_quality_path.exists()
    df = (
        pd.read_csv(out, dtype=str)
        .fillna("")
        .sort_values("uniprot_id")
        .reset_index(drop=True)
    )
    cols = output_columns(False)
    expected_cols = [*cols, "orthologs_json", "orthologs_count"]
    assert list(df.columns) == expected_cols
    assert list(df["uniprot_id"]) == ["P00000", "P12345"]
    assert list(df["orthologs_count"]) == ["0", "1"]
    assert "Q9XYZ1" in df.iloc[1]["orthologs_json"]
    iso_df = pd.read_csv(iso_out, dtype=str).fillna("")
    assert list(iso_df.columns) == [
        "parent_uniprot_id",
        "isoform_uniprot_id",
        "isoform_name",
        "isoform_synonyms",
        "is_canonical",
    ]
    assert len(iso_df) == 2
    orth_df = pd.read_csv(orth_out, dtype=str).fillna("")
    assert list(orth_df.columns) == [
        "source_uniprot_id",
        "source_ensembl_gene_id",
        "source_species",
        "target_species",
        "target_gene_symbol",
        "target_ensembl_gene_id",
        "target_uniprot_id",
        "homology_type",
        "perc_id",
        "perc_pos",
        "dn",
        "ds",
        "is_high_confidence",
        "source_db",
    ]
    assert len(orth_df) == 1
    metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    assert metadata["rows"] == 2
    assert metadata["columns"] == len(expected_cols)
    assert metadata["config"]["input"] == str(inp)
    iso_metadata = yaml.safe_load(iso_meta_path.read_text(encoding="utf-8"))
    assert iso_metadata["rows"] == len(iso_df)
    assert iso_metadata["columns"] == len(iso_df.columns)
    orth_metadata = yaml.safe_load(orth_meta_path.read_text(encoding="utf-8"))
    assert orth_metadata["rows"] == len(orth_df)
    assert orth_metadata["columns"] == len(orth_df.columns)
    assert orth_metadata["config"]["with_orthologs"] is True
