from __future__ import annotations

import json
import requests_mock

from typing import Sequence, cast

from chembl_targets import TargetConfig, fetch_targets, normalise_ids


def test_normalise_ids() -> None:
    raw: list[str | None] = [" CHEMBL1 ", "chembl1", "", None, "CHEMBL2"]
    assert normalise_ids(cast(Sequence[str], raw)) == ["CHEMBL1", "CHEMBL2"]


def test_fetch_targets_parses_fields(requests_mock: requests_mock.Mocker) -> None:
    cfg = TargetConfig(base_url="http://test", list_format="json", rps=0)
    url = "http://test/target/CHEMBL612?format=json"
    payload = {
        "pref_name": "Example",
        "target_type": "SINGLE PROTEIN",
        "organism": "Homo sapiens",
        "tax_id": 9606,
        "species_group_flag": 0,
        "target_components": [
            {
                "component_id": 1,
                "accession": "P12345",
                "component_type": "PROTEIN",
                "component_description": "desc",
                "target_component_synonyms": [
                    {"component_synonym": "ABC1", "syn_type": "GENE_SYMBOL"},
                    {"component_synonym": "ProtAlt", "syn_type": "PROTEIN_NAME"},
                    {"component_synonym": "1.1.1.1", "syn_type": "EC_NUMBER"},
                    {"component_synonym": "AltUni", "syn_type": "UNIPROT"},
                ],
                "target_component_xrefs": [
                    {"xref_src_db": "Ensembl", "xref_id": "ENSG000001"},
                    {"xref_src_db": "UniProt", "xref_id": "P12345"},
                    {
                        "xref_src_db": "HGNC",
                        "xref_id": "HGNC:5",
                        "xref_name": "HGNCNAME",
                    },
                ],
            }
        ],
        "cross_references": [
            {"xref_db": "IUPHAR/BPS", "xref_id": "123"},
            {"xref_db": "UniProt", "xref_id": "Q99999"},
        ],
        "protein_classification": {
            "pref_name": "L5",
            "parent": {"pref_name": "L4"},
        },
    }
    requests_mock.get(url, json=payload)
    df = fetch_targets(["CHEMBL612"], cfg)
    assert list(df.columns) == [
        "target_chembl_id",
        "pref_name",
        "protein_name_canonical",
        "target_type",
        "organism",
        "tax_id",
        "species_group_flag",
        "target_components",
        "protein_classifications",
        "cross_references",
        "gene_symbol_list",
        "protein_synonym_list",
        "ec_number_list",
        "chembl_alternative_name_list",
        "uniprot_id_list",
        "hgnc_name",
        "hgnc_id",
    ]
    record = df.iloc[0]
    comps = json.loads(record["target_components"])
    assert comps[0]["accession"] == "P12345"
    refs = json.loads(record["cross_references"])
    assert [r["xref_db"] for r in refs] == [
        "Ensembl",
        "HGNC",
        "IUPHAR/BPS",
        "UniProt",
        "UniProt",
    ]
    genes = json.loads(record["gene_symbol_list"])
    assert genes == ["ABC1"]
    names = json.loads(record["protein_synonym_list"])
    assert names == ["ProtAlt"]
    ec_numbers = json.loads(record["ec_number_list"])
    assert ec_numbers == ["1.1.1.1"]
    alt_names = json.loads(record["chembl_alternative_name_list"])
    assert alt_names == ["AltUni"]
    uniprot_ids = json.loads(record["uniprot_id_list"])
    assert uniprot_ids == ["P12345", "Q99999"]
    assert record["hgnc_name"] == "HGNCNAME"
    assert record["hgnc_id"] == "5"
    classes = json.loads(record["protein_classifications"])
    assert classes == ["L4", "L5"]
