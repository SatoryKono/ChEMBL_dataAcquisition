from __future__ import annotations
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests_mock

from uniprot_normalize import normalize_entry, output_columns

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "get_uniprot_target_data.py"


def _load_sample() -> dict:
    return json.loads((ROOT / "tests/data/uniprot_entry.json").read_text())


def test_normalize_entry_sorts_lists() -> None:
    entry = _load_sample()
    data = normalize_entry(entry, include_sequence=False)
    assert data["protein_alternative_names"] == ["Alt A", "Alt B"]
    assert data["isoform_ids"] == ["P12345-1", "P12345-2"]
    assert data["domains_pfam"] == [("PF00001", "DomainA"), ("PF00002", "DomainB")]
    assert data["features_signal_peptide"] == ["1-10"]
    assert data["features_transmembrane"] == ["20-40"]
    assert data["ptm_modified_residue"] == ["150:Phosphoserine"]


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
    with requests_mock.Mocker() as m:
        m.get(
            "https://rest.uniprot.org/uniprotkb/search",
            [
                {"json": {"results": [sample]}},
                {"json": {"results": []}},
            ],
        )
        module.main(
            [
                "--input",
                str(inp),
                "--output",
                str(out),
                "--sep",
                ",",
                "--encoding",
                "utf-8",
            ]
        )
    df = pd.read_csv(out, dtype=str).fillna("")
    cols = output_columns(False)
    assert list(df.columns) == cols
    assert len(df) == 2
    assert df.loc[0, "uniprot_id"] == "P12345"
    assert df.loc[1, "uniprot_id"] == "P00000"
