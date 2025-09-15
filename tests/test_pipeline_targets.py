from __future__ import annotations

import json
from typing import Any, Dict, List
from pathlib import Path

import pandas as pd
import pytest

from chembl_targets import TargetConfig
from hgnc_client import HGNCRecord
from pipeline_targets import PipelineConfig, run_pipeline


def make_uniprot(
    accession: str,
    organism: str,
    taxon: int,
    lineage: List[str],
    *,
    hgnc: str | None = None,
) -> Dict[str, Any]:
    return {
        "primaryAccession": accession,
        "entryType": "reviewed",
        "proteinDescription": {
            "recommendedName": {"fullName": {"value": f"Protein {accession}"}}
        },
        "genes": [{"geneName": {"value": f"GENE{accession}"}}],
        "organism": {
            "scientificName": organism,
            "taxonId": taxon,
            "lineage": lineage,
        },
        "sequence": {"length": 100, "sequenceChecksum": "abcd"},
        "uniProtKBCrossReferences": [
            {"database": "ChEMBL", "id": "CHEMBL1"},
            {"database": "Ensembl", "id": "ENSG1"},
            {"database": "PDB", "id": "1ABC"},
            {"database": "AlphaFoldDB", "id": accession},
        ]
        + ([{"database": "HGNC", "id": hgnc}] if hgnc else []),
        "entryAudit": {
            "lastAnnotationUpdateDate": "2024-01-01",
            "entryVersion": 2,
        },
    }


class DummyUniProt:
    def __init__(self, records: Dict[str, Dict[str, Any]]):
        self.records = records

    def fetch(self, acc: str) -> Dict[str, Any] | None:  # pragma: no cover - simple
        return self.records.get(acc)


class DummyHGNC:
    def __init__(self, data: Dict[str, HGNCRecord]):
        self.data = data

    def fetch(self, acc: str) -> HGNCRecord:
        return self.data.get(acc, HGNCRecord(acc, "", "", "", ""))


class DummyGtoP:
    def __init__(self, endpoints: Dict[tuple[int, str], List[Any]]):
        self.endpoints = endpoints

    def fetch_target_endpoint(
        self,
        target_id: int,
        endpoint: str,
        params: Dict[str, Any] | None = None,
    ) -> List[Any]:  # pragma: no cover - simple
        return self.endpoints.get((target_id, endpoint), [])


def fake_resolve(client: DummyGtoP, identifier: str, id_column: str) -> Dict[str, Any]:
    return {"targetId": 111, "name": "T", "species": "Human"}


def make_chembl_df(accessions: List[str]) -> pd.DataFrame:
    comps = [
        {
            "component_id": i + 1,
            "accession": acc,
            "component_type": "protein",
            "component_description": "desc",
        }
        for i, acc in enumerate(accessions)
    ]
    return pd.DataFrame(
        [
            {
                "target_chembl_id": "CHEMBL1",
                "pref_name": "Pref",
                "target_type": "SINGLE PROTEIN",
                "organism": "Homo sapiens",
                "target_components": json.dumps(comps, sort_keys=True),
                "protein_classifications": json.dumps(["ClassA"], sort_keys=True),
                "cross_references": json.dumps([], sort_keys=True),
            }
        ]
    )


def test_pipeline_single_target(monkeypatch: pytest.MonkeyPatch) -> None:
    def chembl_fetch(ids: List[str], cfg: TargetConfig | None = None) -> pd.DataFrame:
        return make_chembl_df(["P12345"])

    uni = DummyUniProt(
        {
            "P12345": make_uniprot(
                "P12345",
                "Homo sapiens",
                9606,
                ["Eukaryota", "Chordata", "Mammalia", "Primates", "Hominidae"],
                hgnc="HGNC:1",
            )
        }
    )
    hgnc = DummyHGNC({"P12345": HGNCRecord("P12345", "HGNC:1", "SYMB", "Name", "Prot")})
    gtop = DummyGtoP(
        {
            (111, "synonyms"): [{"synonym": "Syn1"}],
            (111, "naturalLigands"): [1, 2],
            (111, "interactions"): [{"ligandId": 1}, {"ligandId": 2}],
            (111, "function"): [{"functionText": "func"}],
        }
    )
    monkeypatch.setattr("pipeline_targets.resolve_target", fake_resolve)
    cfg = PipelineConfig()
    df = run_pipeline(
        ["CHEMBL1"],
        cfg,
        chembl_fetcher=chembl_fetch,
        uniprot_client=uni,
        hgnc_client=hgnc,
        gtop_client=gtop,
    )
    row = df.iloc[0]
    assert row["uniprot_id_primary"] == "P12345"
    assert row["hgnc_id"] == "HGNC:1"
    assert row["gtop_natural_ligands_n"] == 2
    assert row["gtop_interactions_n"] == 2
    assert json.loads(row["gtop_synonyms"]) == ["Syn1"]
    names_all = row["names_all"].split("|")
    assert "Pref" in names_all and "Protein P12345" in names_all
    syns_all = row["synonyms_all"].split("|")
    assert "Syn1" in syns_all


def test_pipeline_selects_human_uniprot(monkeypatch: pytest.MonkeyPatch) -> None:
    def chembl_fetch(ids: List[str], cfg: TargetConfig | None = None) -> pd.DataFrame:
        return make_chembl_df(["Q11111", "Q22222"])

    uni = DummyUniProt(
        {
            "Q11111": make_uniprot(
                "Q11111", "Mus musculus", 10090, ["Eukaryota"], hgnc=None
            ),
            "Q22222": make_uniprot(
                "Q22222", "Homo sapiens", 9606, ["Eukaryota"], hgnc="HGNC:2"
            ),
        }
    )
    hgnc = DummyHGNC({"Q22222": HGNCRecord("Q22222", "HGNC:2", "GSYM", "Name", "Prot")})
    gtop = DummyGtoP({})
    monkeypatch.setattr("pipeline_targets.resolve_target", fake_resolve)
    cfg = PipelineConfig()
    df = run_pipeline(
        ["CHEMBL1"],
        cfg,
        chembl_fetcher=chembl_fetch,
        uniprot_client=uni,
        hgnc_client=hgnc,
        gtop_client=gtop,
    )
    row = df.iloc[0]
    assert row["uniprot_id_primary"] == "Q22222"
    ids = json.loads(row["uniprot_ids_all"])
    assert ids == ["Q11111", "Q22222"]


def test_pipeline_missing_hgnc(monkeypatch: pytest.MonkeyPatch) -> None:
    def chembl_fetch(ids: List[str], cfg: TargetConfig | None = None) -> pd.DataFrame:
        return make_chembl_df(["P12345"])

    uni = DummyUniProt(
        {
            "P12345": make_uniprot(
                "P12345", "Homo sapiens", 9606, ["Eukaryota"], hgnc=None
            )
        }
    )
    hgnc = DummyHGNC({})
    gtop = DummyGtoP({})
    monkeypatch.setattr("pipeline_targets.resolve_target", fake_resolve)
    cfg = PipelineConfig()
    df = run_pipeline(
        ["CHEMBL1"],
        cfg,
        chembl_fetcher=chembl_fetch,
        uniprot_client=uni,
        hgnc_client=hgnc,
        gtop_client=gtop,
    )
    row = df.iloc[0]
    assert row["hgnc_id"] == ""
    assert row["gene_symbol"].startswith("GENE")


def test_pipeline_reproducible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def chembl_fetch(ids: List[str], cfg: TargetConfig | None = None) -> pd.DataFrame:
        return make_chembl_df(["P12345"])

    uni = DummyUniProt(
        {
            "P12345": make_uniprot(
                "P12345", "Homo sapiens", 9606, ["Eukaryota"], hgnc="HGNC:1"
            )
        }
    )
    hgnc = DummyHGNC({"P12345": HGNCRecord("P12345", "HGNC:1", "SYMB", "Name", "Prot")})
    gtop = DummyGtoP({})
    monkeypatch.setattr("pipeline_targets.resolve_target", fake_resolve)
    cfg = PipelineConfig()
    out1 = run_pipeline(
        ["CHEMBL1"],
        cfg,
        chembl_fetcher=chembl_fetch,
        uniprot_client=uni,
        hgnc_client=hgnc,
        gtop_client=gtop,
    )
    out2 = run_pipeline(
        ["CHEMBL1"],
        cfg,
        chembl_fetcher=chembl_fetch,
        uniprot_client=uni,
        hgnc_client=hgnc,
        gtop_client=gtop,
    )
    p1 = tmp_path / "out1.csv"
    p2 = tmp_path / "out2.csv"
    out1.to_csv(p1, index=False)
    out2.to_csv(p2, index=False)
    assert p1.read_bytes() == p2.read_bytes()


def test_pipeline_respects_config_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline uses column order defined in configuration."""

    def chembl_fetch(ids: List[str], cfg: TargetConfig | None = None) -> pd.DataFrame:
        return make_chembl_df(["P12345"])

    uni = DummyUniProt(
        {
            "P12345": make_uniprot(
                "P12345", "Homo sapiens", 9606, ["Eukaryota"], hgnc=None
            )
        }
    )
    hgnc = DummyHGNC({})
    gtop = DummyGtoP({})
    monkeypatch.setattr("pipeline_targets.resolve_target", fake_resolve)
    cfg = PipelineConfig(columns=["target_chembl_id", "uniprot_id_primary"])
    df = run_pipeline(
        ["CHEMBL1"],
        cfg,
        chembl_fetcher=chembl_fetch,
        uniprot_client=uni,
        hgnc_client=hgnc,
        gtop_client=gtop,
    )
    assert list(df.columns) == ["target_chembl_id", "uniprot_id_primary"]
