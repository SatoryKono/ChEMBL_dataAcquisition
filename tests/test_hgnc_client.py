from __future__ import annotations

from pathlib import Path

import pandas as pd

from hgnc_client import map_uniprot_to_hgnc

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"


def _write_csv(path: Path, values: list[str]) -> None:
    df = pd.DataFrame({"uniprot_id": values})
    df.to_csv(path, index=False)


def _read_output(path: Path) -> list[dict[str, str]]:
    df = pd.read_csv(path).fillna("")
    return df.to_dict(orient="records")


def test_valid_uniprot_id(requests_mock, tmp_path: Path) -> None:
    in_csv = tmp_path / "in.csv"
    _write_csv(in_csv, ["P35348"])
    out_csv = tmp_path / "out.csv"

    hgnc_url = "https://rest.genenames.org/fetch/uniprot_ids/P35348"
    requests_mock.get(
        hgnc_url,
        json={
            "response": {
                "docs": [
                    {
                        "symbol": "ADRA1A",
                        "name": "adrenoceptor alpha 1A",
                        "hgnc_id": "HGNC:277",
                    }
                ]
            }
        },
    )
    uniprot_url = "https://rest.uniprot.org/uniprotkb/P35348.json"
    requests_mock.get(
        uniprot_url,
        json={
            "proteinDescription": {
                "recommendedName": {
                    "fullName": {"value": "Alpha-1A adrenergic receptor"}
                }
            }
        },
    )

    map_uniprot_to_hgnc(in_csv, out_csv, CONFIG, config_section="hgnc")
    assert _read_output(out_csv) == [
        {
            "uniprot_id": "P35348",
            "hgnc_id": "HGNC:277",
            "gene_symbol": "ADRA1A",
            "gene_name": "adrenoceptor alpha 1A",
            "protein_name": "Alpha-1A adrenergic receptor",
        }
    ]


def test_non_human_uniprot_id(requests_mock, tmp_path: Path) -> None:
    in_csv = tmp_path / "in.csv"
    _write_csv(in_csv, ["Q91X72"])
    out_csv = tmp_path / "out.csv"

    hgnc_url = "https://rest.genenames.org/fetch/uniprot_ids/Q91X72"
    requests_mock.get(hgnc_url, json={"response": {"docs": []}})
    uniprot_url = "https://rest.uniprot.org/uniprotkb/Q91X72.json"
    uniprot_mock = requests_mock.get(uniprot_url, json={})

    map_uniprot_to_hgnc(in_csv, out_csv, CONFIG, config_section="hgnc")
    assert _read_output(out_csv) == [
        {
            "uniprot_id": "Q91X72",
            "hgnc_id": "",
            "gene_symbol": "",
            "gene_name": "",
            "protein_name": "",
        }
    ]
    assert uniprot_mock.call_count == 0


def test_duplicate_input_ids(requests_mock, tmp_path: Path) -> None:
    in_csv = tmp_path / "in.csv"
    _write_csv(in_csv, ["P35348", "P35348"])
    out_csv = tmp_path / "out.csv"

    hgnc_url = "https://rest.genenames.org/fetch/uniprot_ids/P35348"
    hgnc_mock = requests_mock.get(
        hgnc_url,
        json={
            "response": {
                "docs": [
                    {
                        "symbol": "ADRA1A",
                        "name": "adrenoceptor alpha 1A",
                        "hgnc_id": "HGNC:277",
                    }
                ]
            }
        },
    )
    uniprot_url = "https://rest.uniprot.org/uniprotkb/P35348.json"
    uniprot_mock = requests_mock.get(
        uniprot_url,
        json={
            "proteinDescription": {
                "recommendedName": {
                    "fullName": {"value": "Alpha-1A adrenergic receptor"}
                }
            }
        },
    )

    map_uniprot_to_hgnc(in_csv, out_csv, CONFIG, config_section="hgnc")
    assert _read_output(out_csv) == [
        {
            "uniprot_id": "P35348",
            "hgnc_id": "HGNC:277",
            "gene_symbol": "ADRA1A",
            "gene_name": "adrenoceptor alpha 1A",
            "protein_name": "Alpha-1A adrenergic receptor",
        },
        {
            "uniprot_id": "P35348",
            "hgnc_id": "HGNC:277",
            "gene_symbol": "ADRA1A",
            "gene_name": "adrenoceptor alpha 1A",
            "protein_name": "Alpha-1A adrenergic receptor",
        },
    ]
    assert hgnc_mock.call_count == 1
    assert uniprot_mock.call_count == 1
