from __future__ import annotations

import importlib
from typing import Iterable, List

import requests  # type: ignore[import-untyped]
import pandas as pd
import requests_mock as requests_mock_lib

get_testitems = importlib.import_module("library.chembl_library").get_testitems
add_pubchem_data = importlib.import_module("library.testitem_library").add_pubchem_data
testitem_library = importlib.import_module("library.testitem_library")
PUBCHEM_BASE_URL = testitem_library.PUBCHEM_BASE_URL
PUBCHEM_PROPERTIES = testitem_library.PUBCHEM_PROPERTIES
normalize_testitems = importlib.import_module(
    "library.normalize_testitems"
).normalize_testitems


def test_get_testitems_batches_requests() -> None:
    calls: List[List[str]] = []

    class DummyClient:
        def fetch_many_molecules(self, values: Iterable[str]) -> List[dict[str, str]]:
            batch = list(values)
            calls.append(batch)
            return [
                {
                    "molecule_chembl_id": molecule_id,
                    "pref_name": f"Name {molecule_id}",
                }
                for molecule_id in batch
            ]

    df = get_testitems(DummyClient(), ["CHEMBL1", "CHEMBL2", "CHEMBL3"], chunk_size=2)
    assert list(df["molecule_chembl_id"]) == ["CHEMBL1", "CHEMBL2", "CHEMBL3"]
    assert calls == [["CHEMBL1", "CHEMBL2"], ["CHEMBL3"]]


def test_add_pubchem_data_enriches_dataframe(
    requests_mock: requests_mock_lib.Mocker,
) -> None:
    df = pd.DataFrame(
        [
            {"molecule_chembl_id": "CHEMBL1", "canonical_smiles": "C"},
            {"molecule_chembl_id": "CHEMBL2", "canonical_smiles": "C"},
            {"molecule_chembl_id": "CHEMBL3", "canonical_smiles": "CC"},
        ]
    )

    def _pubchem_properties_response(formula: str, cid: int) -> dict[str, object]:
        return {
            "PropertyTable": {
                "Properties": [
                    {
                        "MolecularFormula": formula,
                        "MolecularWeight": 42.0 + cid,
                        "TPSA": 12.3,
                        "XLogP": -1.2,
                        "HBondDonorCount": 1,
                        "HBondAcceptorCount": 2,
                        "RotatableBondCount": 0,
                    }
                ]
            }
        }

    property_suffix = ",".join(prop for prop in PUBCHEM_PROPERTIES if prop != "CID")
    requests_mock.get(
        f"{PUBCHEM_BASE_URL}/compound/smiles/C/property/{property_suffix}/JSON",
        json=_pubchem_properties_response("CH4", 10),
    )
    requests_mock.get(
        f"{PUBCHEM_BASE_URL}/compound/smiles/CC/property/{property_suffix}/JSON",
        json=_pubchem_properties_response("C2H6", 20),
    )
    requests_mock.get(
        f"{PUBCHEM_BASE_URL}/compound/smiles/C/cids/JSON",
        json={"IdentifierList": {"CID": [10]}},
    )
    requests_mock.get(
        f"{PUBCHEM_BASE_URL}/compound/smiles/CC/cids/JSON",
        json={"IdentifierList": {"CID": [20]}},
    )

    enriched = add_pubchem_data(df)
    assert requests_mock.call_count == 4

    assert enriched.loc[0, "pubchem_cid"] == 10
    assert enriched.loc[1, "pubchem_cid"] == 10  # cached duplicate
    assert enriched.loc[2, "pubchem_cid"] == 20
    assert enriched.loc[2, "pubchem_molecular_formula"] == "C2H6"


def test_add_pubchem_data_handles_invalid_json(
    requests_mock: requests_mock_lib.Mocker,
) -> None:
    df = pd.DataFrame(
        [
            {"molecule_chembl_id": "CHEMBL1", "canonical_smiles": "C"},
        ]
    )

    property_suffix = ",".join(prop for prop in PUBCHEM_PROPERTIES if prop != "CID")
    requests_mock.get(
        f"{PUBCHEM_BASE_URL}/compound/smiles/C/property/{property_suffix}/JSON",
        text="not-json",
        status_code=200,
    )
    requests_mock.get(
        f"{PUBCHEM_BASE_URL}/compound/smiles/C/cids/JSON",
        json={"IdentifierList": {"CID": []}},
    )

    enriched = add_pubchem_data(df, http_client_config={"max_retries": 1, "rps": 0.0})

    assert requests_mock.call_count == 2
    assert pd.isna(enriched.loc[0, "pubchem_cid"])


def test_add_pubchem_data_handles_network_errors(
    requests_mock: requests_mock_lib.Mocker,
) -> None:
    df = pd.DataFrame(
        [
            {"molecule_chembl_id": "CHEMBL1", "canonical_smiles": "C"},
        ]
    )

    property_suffix = ",".join(prop for prop in PUBCHEM_PROPERTIES if prop != "CID")
    requests_mock.get(
        f"{PUBCHEM_BASE_URL}/compound/smiles/C/property/{property_suffix}/JSON",
        exc=requests.exceptions.ConnectTimeout,
    )
    requests_mock.get(
        f"{PUBCHEM_BASE_URL}/compound/smiles/C/cids/JSON",
        json={"IdentifierList": {"CID": []}},
    )

    enriched = add_pubchem_data(df, http_client_config={"max_retries": 1, "rps": 0.0})

    assert requests_mock.call_count == 2
    assert pd.isna(enriched.loc[0, "pubchem_cid"])


def test_add_pubchem_data_fetches_cid_after_property_failure(
    requests_mock: requests_mock_lib.Mocker,
) -> None:
    df = pd.DataFrame(
        [
            {"molecule_chembl_id": "CHEMBL1", "canonical_smiles": "C"},
        ]
    )

    requests_mock.get(
        f"{PUBCHEM_BASE_URL}/compound/smiles/C/property/"
        "CID,MolecularFormula,MolecularWeight,TPSA,XLogP,HBondDonorCount,HBondAcceptorCount,RotatableBondCount/JSON",
        status_code=500,
        json={"Fault": "error"},
    )
    requests_mock.get(
        f"{PUBCHEM_BASE_URL}/compound/smiles/C/cids/JSON",
        json={"IdentifierList": {"CID": [123]}},
    )

    enriched = add_pubchem_data(df, http_client_config={"max_retries": 1, "rps": 0.0})

    assert requests_mock.call_count == 2
    assert enriched.loc[0, "pubchem_cid"] == 123
    assert pd.isna(enriched.loc[0, "pubchem_molecular_formula"])


def test_add_pubchem_data_handles_missing_smiles() -> None:
    df = pd.DataFrame([{"molecule_chembl_id": "CHEMBL1"}])
    result = add_pubchem_data(df)
    assert result is df


def test_normalize_testitems_flattens_payload() -> None:
    raw = pd.DataFrame(
        [
            {
                "molecule_chembl_id": "chembl1",
                "pref_name": " Compound ",
                "molecule_type": "Small molecule",
                "max_phase": "2",
                "black_box_warning": "false",
                "molecule_structures": {
                    "canonical_smiles": " C ",
                    "standard_inchi": " InChI ",
                    "standard_inchi_key": " KEY ",
                },
                "molecule_properties": {
                    "full_mwt": "120.5",
                    "num_ro5_violations": "0",
                    "molecular_species": "NEUTRAL",
                },
                "molecule_synonyms": [
                    {"synonyms": "Alias"},
                    {"synonyms": "Alias "},
                ],
                "atc_classifications": ["A01", "A01"],
                "cross_references": [
                    {
                        "xref_id": "X1",
                        "xref_src": "SRC",
                        "xref_name": "Name",
                    },
                    {"xref_id": "", "xref_src": "", "xref_name": ""},
                ],
                "molecule_hierarchy": {
                    "molecule_chembl_id": "CHEMBL1",
                    "parent_chembl_id": "CHEMBL0",
                    "active_chembl_id": "CHEMBL3",
                },
            }
        ]
    )

    normalised = normalize_testitems(raw)
    assert normalised.loc[0, "molecule_chembl_id"] == "CHEMBL1"
    assert normalised.loc[0, "pref_name"] == "Compound"
    assert normalised.loc[0, "canonical_smiles"] == "C"
    assert normalised.loc[0, "standard_inchi_key"] == "KEY"
    assert normalised.loc[0, "chembl_full_mwt"] == 120.5
    assert normalised.loc[0, "chembl_num_ro5_violations"] == 0
    assert normalised.loc[0, "chembl_molecular_species"] == "NEUTRAL"
    assert normalised.loc[0, "synonyms"] == ["Alias"]
    assert normalised.loc[0, "atc_classifications"] == ["A01"]
    assert normalised.loc[0, "cross_references"] == [
        {"xref_id": "X1", "xref_src": "SRC", "xref_name": "Name"}
    ]
    assert normalised.loc[0, "parent_chembl_id"] == "CHEMBL0"
    assert normalised.loc[0, "salt_chembl_id"] == "CHEMBL1"
    assert normalised.loc[0, "active_chembl_id"] == "CHEMBL3"
