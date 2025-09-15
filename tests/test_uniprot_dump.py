from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd
import requests_mock

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


def test_cli_writes_output(tmp_path: Path) -> None:
    from importlib.machinery import SourceFileLoader

    module = SourceFileLoader("get_uniprot_target_data", str(SCRIPT_PATH)).load_module()

    inp = tmp_path / "inp.csv"
    inp.write_text("uniprot_id\nP12345\np12345\nP00000\n")
    out = tmp_path / "out.csv"

    sample = _load_sample()
    iso_out = tmp_path / "iso.csv"
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
                "--isoforms-output",
                str(iso_out),
            ]
        )
    df = (
        pd.read_csv(out, dtype=str)
        .fillna("")
        .sort_values("uniprot_id")
        .reset_index(drop=True)
    )
    cols = output_columns(False)
    assert list(df.columns) == cols
    assert list(df["uniprot_id"]) == ["P00000", "P12345"]
    iso_df = pd.read_csv(iso_out, dtype=str).fillna("")
    assert list(iso_df.columns) == [
        "parent_uniprot_id",
        "isoform_uniprot_id",
        "isoform_name",
        "isoform_synonyms",
        "is_canonical",
    ]
    assert len(iso_df) == 2
