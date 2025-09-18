from pathlib import Path
from typing import Dict, Iterable

import json
import sys

import pandas as pd

sys.path.insert(0, str(Path("scripts")))
from pipeline_targets_main import (
    add_iuphar_classification,
    add_protein_classification,
    add_activity_fields,
    add_isoform_fields,
    add_uniprot_fields,
    extract_activity,
    extract_isoform,
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


def test_add_protein_classification_fetches_once() -> None:
    samples = json.loads((Path("tests/data/protein_samples.json").read_text()))
    calls: list[list[str]] = []

    def fetcher(accessions: Iterable[str]) -> Dict[str, dict]:
        collected = list(accessions)
        calls.append(collected)
        return {acc: samples["gpcr"] for acc in collected}

    df = pd.DataFrame({"uniprot_id_primary": ["P00000", "", "P00000"]})
    add_protein_classification(df, fetcher)
    assert len(calls) == 1
    assert calls[0] == ["P00000"]


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


def test_extract_activity() -> None:
    entry = {
        "comments": [
            {
                "commentType": "CATALYTIC ACTIVITY",
                "reaction": {
                    "name": {"value": "A + B = C"},
                    "ecNumber": [{"value": "1.1.1.1"}],
                },
            },
            {
                "commentType": "CATALYTIC ACTIVITY",
                "reaction": {
                    "name": {"value": "D = E"},
                    "ecNumbers": [{"value": "2.2.2.2"}, {"value": "3.3.3.3"}],
                },
            },
        ]
    }
    result = extract_activity(entry)
    assert result["reactions"] == "A + B = C|D = E"
    assert result["reaction_ec_numbers"] == "1.1.1.1|2.2.2.2|3.3.3.3"


def test_add_activity_fields() -> None:
    df = pd.DataFrame({"uniprot_id_primary": ["P00001"]})

    def fetch_entry(_: str) -> dict:
        return {
            "comments": [
                {
                    "commentType": "CATALYTIC ACTIVITY",
                    "reaction": {
                        "name": {"value": "X = Y"},
                        "ecNumber": [{"value": "4.4.4.4"}],
                    },
                }
            ]
        }

    out = add_activity_fields(df, fetch_entry)
    row = out.iloc[0]
    assert row["reactions"] == "X = Y"
    assert row["reaction_ec_numbers"] == "4.4.4.4"


def test_extract_isoform() -> None:
    entry = {
        "comments": [
            {
                "commentType": "ALTERNATIVE PRODUCTS",
                "isoforms": [
                    {
                        "name": {"value": "Isoform 1"},
                        "isoformIds": ["P1-1"],
                        "synonyms": [{"value": "Alpha"}, {"value": "Beta"}],
                    },
                    {
                        "name": {"value": "Isoform 2"},
                        "isoformIds": ["P1-2"],
                        "synonyms": [],
                    },
                ],
            }
        ]
    }
    result = extract_isoform(entry)
    assert result["isoform_names"] == "Isoform 1|Isoform 2"
    assert result["isoform_ids"] == "P1-1|P1-2"
    assert result["isoform_synonyms"] == "Alpha:Beta|N/A"


def test_add_isoform_fields() -> None:
    df = pd.DataFrame({"uniprot_id_primary": ["P99999"]})

    def fetch_entry(_: str) -> dict:
        return {
            "comments": [
                {
                    "commentType": "ALTERNATIVE PRODUCTS",
                    "isoforms": [
                        {
                            "name": {"value": "Isoform 1"},
                            "isoformIds": ["P1-1"],
                            "synonyms": [{"value": "Alpha"}],
                        }
                    ],
                }
            ]
        }

    out = add_isoform_fields(df, fetch_entry)
    row = out.iloc[0]
    assert row["isoform_names"] == "Isoform 1"
    assert row["isoform_ids"] == "P1-1"
    assert row["isoform_synonyms"] == "Alpha"
