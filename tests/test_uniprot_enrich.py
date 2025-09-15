import json
from pathlib import Path

import pandas as pd
import pytest
import requests_mock

from uniprot_enrich import enrich_uniprot

MOCK_ENTRY = {
    "primaryAccession": "P12345",
    "uniProtkbId": "TEST_HUMAN",
    "entryType": "Reviewed",
    "secondaryAccessions": ["Q99999"],
    "proteinDescription": {
        "recommendedName": {
            "fullName": {"value": "Test protein"},
            "ecNumbers": [{"value": "1.2.3.4"}],
        },
        "alternativeNames": [
            {"fullName": {"value": "Alt1"}},
            {"fullName": {"value": "Alt2"}},
        ],
    },
    "genes": [{"geneName": {"value": "TP"}}],
    "organism": {
        "taxonId": 9606,
        "lineage": ["Eukaryota", "Metazoa", "Chordata", "Testus"],
    },
    "comments": [
        {
            "commentType": "SUBCELLULAR_LOCATION",
            "subcellularLocations": [
                {
                    "location": {"value": "Cytoplasm"},
                    "topology": {"value": "Peripheral membrane protein"},
                }
            ],
        },
        {
            "commentType": "ALTERNATIVE_PRODUCTS",
            "isoforms": [
                {
                    "name": {"value": "Isoform 1"},
                    "id": "P12345-1",
                    "synonyms": [{"value": "Iso1"}],
                }
            ],
        },
        {
            "commentType": "CATALYTIC_ACTIVITY",
            "reaction": {"name": "A + B = C", "ecNumber": "1.2.3.4"},
        },
    ],
    "features": [
        {
            "type": "Glycosylation",
            "description": "N-linked (GlcNAc) asparagine",
            "location": {"start": {"value": 2}, "end": {"value": 2}},
        },
        {
            "type": "Topological domain",
            "description": "Extracellular",
            "location": {"start": {"value": 1}, "end": {"value": 3}},
        },
        {
            "type": "Transmembrane region",
            "location": {"start": {"value": 10}, "end": {"value": 30}},
        },
        {
            "type": "Intramembrane region",
            "location": {"start": {"value": 40}, "end": {"value": 50}},
        },
        {
            "type": "Modified residue",
            "description": "Phosphothreonine",
            "location": {"start": {"value": 5}, "end": {"value": 5}},
        },
    ],
    "uniProtKBCrossReferences": [
        {
            "database": "HGNC",
            "id": "HGNC:1",
            "properties": [{"key": "Name", "value": "TP"}],
        },
        {"database": "GuidetoPHARMACOLOGY", "id": "G123"},
        {
            "database": "GO",
            "id": "GO:0003677",
            "properties": [{"key": "GoTerm", "value": "F:DNA binding"}],
        },
        {
            "database": "GO",
            "id": "GO:0005634",
            "properties": [{"key": "GoTerm", "value": "C:nucleus"}],
        },
        {"database": "PROSITE", "id": "PS12345"},
        {"database": "Pfam", "id": "PF00001"},
    ],
}

MOCK_SECONDARY = {
    "primaryAccession": "Q99999",
    "proteinDescription": {
        "recommendedName": {"fullName": {"value": "Secondary protein"}}
    },
}


@pytest.fixture()
def data_file(tmp_path: Path) -> Path:
    src = Path("tests/data/uniprot_sample.csv")
    dst = tmp_path / "uniprot_sample.csv"
    dst.write_text(src.read_text())
    return dst


def test_enrich_uniprot(data_file: Path) -> None:
    with requests_mock.Mocker() as m:
        m.get(
            "https://rest.uniprot.org/uniprotkb/P12345?format=json",
            text=json.dumps(MOCK_ENTRY),
        )
        m.get(
            "https://rest.uniprot.org/uniprotkb/Q99999?format=json",
            text=json.dumps(MOCK_SECONDARY),
        )
        m.get(
            "https://rest.uniprot.org/uniprotkb/P99999?format=json",
            status_code=404,
        )
        enrich_uniprot(str(data_file))
        assert m.call_count == 3
    df = pd.read_csv(data_file, dtype=str).fillna("")
    assert df.loc[0, "recommended_name"] == "Test protein"
    assert df.loc[0, "synonyms"] == "Alt1|Alt2"
    assert df.loc[1, "recommended_name"] == ""
    assert df.loc[2, "gene_name"] == "TP"
    assert df.loc[0, "secondary_accession_names"] == "Secondary protein"
    assert df.loc[0, "PROSITE"] == "PS12345"
    assert df.loc[0, "reaction_ec_numbers"] == "1.2.3.4"
    assert df.loc[0, "subcellular_location"] == "Cytoplasm"
    assert "Peripheral membrane protein" in df.loc[0, "topology"]
    assert df.loc[0, "transmembrane"] == "1"
    assert df.loc[0, "intramembrane"] == "1"
    # column order
    expected_cols = ["uniprot_id", "other"] + list(
        enrich_uniprot.__globals__["OUTPUT_COLUMNS"]
    )
    assert list(df.columns) == expected_cols
