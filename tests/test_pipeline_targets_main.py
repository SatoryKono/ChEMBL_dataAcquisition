from pathlib import Path

import json
import sys

import pandas as pd

sys.path.insert(0, str(Path("scripts")))
from pipeline_targets_main import (
    add_iuphar_classification,
    add_protein_classification,
    merge_chembl_fields,
)


def test_merge_chembl_fields_adds_columns():
    pipeline_df = pd.DataFrame(
        {
            "target_chembl_id": ["CHEMBL1"],
            "hgnc_id": ["HGNC:1"],
        }
    )
    chembl_df = pd.DataFrame(
        {
            "target_chembl_id": ["CHEMBL1"],
            "species_group_flag": [1],
            "hgnc_name": ["ABC"],
            "hgnc_id": ["HGNC:1"],
        }
    )
    merged = merge_chembl_fields(pipeline_df, chembl_df)
    assert "species_group_flag" in merged.columns
    assert "hgnc_name" in merged.columns
    assert merged.loc[0, "hgnc_id"] == "HGNC:1"


def test_add_iuphar_classification():
    df = pd.DataFrame(
        {
            "uniprot_id_primary": ["P12345"],
            "gtop_target_id": [""],
            "hgnc_name": ["ABC1"],
            "hgnc_id": ["1"],
            "gene_symbol": ["ABC1"],
            "synonyms_all": ["Alpha|Beta"],
        }
    )
    out = add_iuphar_classification(
        df,
        Path("tests/data/iuphar_target.csv"),
        Path("tests/data/iuphar_family.csv"),
    )
    row = out.iloc[0]
    assert row["iuphar_target_id"] == "0001"
    assert row["iuphar_family_id"] == "F001"
    assert row["iuphar_type"] == "Enzyme.Transferase"


def test_add_protein_classification():
    samples = json.loads((Path("tests/data/protein_samples.json").read_text()))

    def fetcher(_: str) -> dict:
        return samples["gpcr"]

    df = pd.DataFrame({"uniprot_id_primary": ["P00000"]})
    out = add_protein_classification(df, fetcher)
    row = out.iloc[0]
    assert row["protein_class_pred_L1"] == "Receptor: GPCR"
    assert row["protein_class_pred_confidence"] == "high"
