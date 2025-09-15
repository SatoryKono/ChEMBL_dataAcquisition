from pathlib import Path
from typing import Dict, Iterable

import json
import sys

import pandas as pd

sys.path.insert(0, str(Path("scripts")))
from pipeline_targets_main import (
    add_iuphar_classification,
    add_protein_classification,
    add_uniprot_fields,
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

    def fetcher(_: Iterable[str]) -> Dict[str, dict]:
        return {"P00000": samples["gpcr"]}

    df = pd.DataFrame({"uniprot_id_primary": ["P00000"]})
    out = add_protein_classification(df, fetcher)
    row = out.iloc[0]
    assert row["protein_class_pred_L1"] == "Receptor: GPCR"
    assert row["protein_class_pred_confidence"] == "high"


def test_add_uniprot_fields() -> None:
    df = pd.DataFrame({"uniprot_id_primary": ["P12345"]})

    def fetch_all(_: Iterable[str]) -> Dict[str, Dict[str, str]]:
        return {
            "P12345": {
                "uniprotkb_Id": "P12345",
                "secondary_uniprot_id": "Q00001",
                "recommended_name": "Protein X",
                "gene_name": "GENE1",
                "secondary_accession_names": "Name1|Name2",
                "molecular_function": "binding",
            }
        }

    out = add_uniprot_fields(df, fetch_all)
    row = out.iloc[0]
    assert row["uniProtkbId"] == "P12345"
    assert row["secondaryAccessions"] == "Q00001"
    assert row["recommendedName"] == "Protein X"
    assert row["geneName"] == "GENE1"
    assert row["secondaryAccessionNames"] == "Name1|Name2"
    assert row["molecular_function"] == "binding"
