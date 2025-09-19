from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import requests_mock as requests_mock_lib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

read_ids = importlib.import_module("library.io").read_ids
CsvConfig = importlib.import_module("library.io_utils").CsvConfig
chembl_testitems_main = importlib.import_module("scripts.chembl_testitems_main").main
PUBCHEM_PROPERTIES = importlib.import_module(
    "library.testitem_library"
).PUBCHEM_PROPERTIES


def test_read_ids_limit(tmp_path: Path) -> None:
    source = Path(__file__).parent / "data" / "testitems_input.csv"
    target = tmp_path / "ids.csv"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    cfg = CsvConfig(sep=",", encoding="utf-8")
    ids = list(read_ids(target, "molecule_chembl_id", cfg, limit=1))
    assert ids == ["CHEMBL1"]


def test_chembl_testitems_main_dry_run(tmp_path: Path) -> None:
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("molecule_chembl_id\nCHEMBL1\nCHEMBL2\n", encoding="utf-8")

    output_csv = tmp_path / "out.csv"
    exit_code = chembl_testitems_main(
        [
            "--input",
            str(input_csv),
            "--output",
            str(output_csv),
            "--dry-run",
            "--log-level",
            "DEBUG",
        ]
    )

    assert exit_code == 0
    assert not output_csv.exists()


def test_chembl_testitems_main_end_to_end(
    tmp_path: Path, requests_mock: requests_mock_lib.Mocker
) -> None:
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("molecule_chembl_id\nCHEMBL1\nCHEMBL2\n", encoding="utf-8")

    base_url = "https://chembl.mock"
    pubchem_base = "https://pubchem.mock/rest/pug"

    def _chembl_payload(
        identifier: str, smiles: str, inchi_key: str, parent: str | None
    ) -> dict[str, object]:
        payload = {
            "molecule_chembl_id": identifier,
            "pref_name": f"Compound {identifier[-1]}",
            "molecule_type": "Small molecule",
            "molecule_properties": {
                "full_mwt": "120.5",
                "alogp": "1.5",
                "num_ro5_violations": "0",
                "molecular_species": "NEUTRAL",
            },
            "molecule_structures": {
                "canonical_smiles": smiles,
                "standard_inchi": f"InChI=1S/{identifier}",
                "standard_inchi_key": inchi_key,
            },
            "molecule_synonyms": [{"synonyms": f"Synonym {identifier}"}],
            "molecule_hierarchy": {
                "molecule_chembl_id": identifier,
                "parent_chembl_id": parent or identifier,
            },
        }
        return payload

    requests_mock.get(
        f"{base_url}/molecule/CHEMBL1.json",
        json=_chembl_payload("CHEMBL1", "C", "KEYONE", None),
    )
    requests_mock.get(
        f"{base_url}/molecule/CHEMBL2.json",
        json=_chembl_payload("CHEMBL2", "CC", "KEYTWO", "CHEMBL1"),
    )

    def _pubchem_response(cid: int, formula: str) -> dict[str, object]:
        return {
            "PropertyTable": {
                "Properties": [
                    {
                        "CID": cid,
                        "MolecularFormula": formula,
                        "MolecularWeight": 50.0 + cid,
                        "TPSA": 10.0,
                        "XLogP": -0.5,
                        "HBondDonorCount": 1,
                        "HBondAcceptorCount": 1,
                        "RotatableBondCount": 0,
                    }
                ]
            }
        }

    requests_mock.get(
        f"{pubchem_base}/compound/smiles/C/property/"
        "MolecularFormula,MolecularWeight,TPSA,XLogP,HBondDonorCount,HBondAcceptorCount,RotatableBondCount/JSON",
        json=_pubchem_response(11, "CH4"),
    )
    requests_mock.get(
        f"{pubchem_base}/compound/smiles/CC/property/"
        "MolecularFormula,MolecularWeight,TPSA,XLogP,HBondDonorCount,HBondAcceptorCount,RotatableBondCount/JSON",
        json=_pubchem_response(22, "C2H6"),
    )
    requests_mock.get(
        f"{pubchem_base}/compound/smiles/C/cids/JSON",
        json={"IdentifierList": {"CID": [11]}},
    )
    requests_mock.get(
        f"{pubchem_base}/compound/smiles/CC/cids/JSON",
        json={"IdentifierList": {"CID": [22]}},
    )

    output_csv = tmp_path / "out.csv"
    exit_code = chembl_testitems_main(
        [
            "--input",
            str(input_csv),
            "--output",
            str(output_csv),
            "--base-url",
            base_url,
            "--chunk-size",
            "1",
            "--pubchem-base-url",
            pubchem_base,
            "--pubchem-timeout",
            "5",
            "--log-level",
            "DEBUG",
        ]
    )

    assert exit_code == 0
    assert output_csv.exists()
    df = pd.read_csv(output_csv)
    assert sorted(df["molecule_chembl_id"].tolist()) == ["CHEMBL1", "CHEMBL2"]
    assert "pubchem_cid" in df.columns
    assert (
        int(df.loc[df["molecule_chembl_id"] == "CHEMBL2", "pubchem_cid"].iloc[0]) == 22
    )
    assert (
        df.loc[df["molecule_chembl_id"] == "CHEMBL2", "salt_chembl_id"].iloc[0]
        == "CHEMBL2"
    )

    meta_file = output_csv.with_name(f"{output_csv.name}.meta.yaml")
    assert meta_file.exists()

    base_path = output_csv.with_name(output_csv.stem)
    quality_report = Path(f"{base_path}_quality_report_table.csv")
    assert quality_report.exists()
    corr_report = Path(f"{base_path}_data_correlation_report_table.csv")
    assert corr_report.exists()
