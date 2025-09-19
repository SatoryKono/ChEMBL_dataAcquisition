from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import pytest
import requests

from hgnc_client import HGNCRecord
from pipeline_targets import (
    DEFAULT_COLUMNS,
    PipelineConfig,
    load_pipeline_config,
    run_pipeline,
)

sys.path.insert(0, str(Path("scripts")))
from pipeline_targets_main import add_protein_classification


def make_uniprot(
    accession: str,
    organism: str,
    taxon: int,
    lineage: List[str],
    *,
    hgnc: str | None = None,
    include_isoform_comment: bool = False,
) -> Dict:
    entry = {
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
    if include_isoform_comment:
        entry["comments"] = [
            {
                "commentType": "ALTERNATIVE_PRODUCTS",
                "isoforms": [
                    {
                        "name": {"value": f"Isoform {accession}-1"},
                        "synonyms": [{"value": f"Iso {accession}"}],
                        "id": f"{accession}-1",
                        "isSequenceDisplayed": True,
                    }
                ],
            }
        ]
    return entry


class DummyUniProt:
    def __init__(self, records: Dict[str, Dict]):
        self.records = records
        self.fetch_calls: List[str] = []
        self.batch_calls: List[List[str]] = []
        self.entry_calls: List[str] = []
        self.isoform_calls: List[str] = []

    def fetch(self, acc: str) -> Dict | None:  # pragma: no cover - simple
        self.fetch_calls.append(acc)
        return self.records.get(acc)

    def fetch_entry_json(self, acc: str) -> Dict | None:  # pragma: no cover - simple
        self.entry_calls.append(acc)
        return self.records.get(acc)

    def fetch_entries_json(
        self, accessions: Iterable[str], *, batch_size: int = 100
    ) -> Dict[str, Dict]:
        chunk = [acc for acc in accessions]
        self.batch_calls.append(chunk)
        return {acc: self.records[acc] for acc in chunk if acc in self.records}

    def fetch_isoforms_fasta(self, acc: str) -> List[str]:  # pragma: no cover - simple
        self.isoform_calls.append(acc)
        return []


class DummyHGNC:
    def __init__(self, data: Dict[str, HGNCRecord]):
        self.data = data

    def fetch(self, acc: str) -> HGNCRecord:
        return self.data.get(acc, HGNCRecord(acc, "", "", "", ""))


class DummyGtoP:
    def __init__(self, endpoints: Dict[tuple[int, str], List[Dict]]):
        self.endpoints = endpoints

    def fetch_target_endpoint(
        self, target_id: int, endpoint: str, params=None
    ):  # pragma: no cover - simple
        return self.endpoints.get((target_id, endpoint), [])


def fake_resolve(client: DummyGtoP, identifier: str, id_column: str):
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


def test_load_pipeline_config_handles_null_sections(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
pipeline:
  columns: null
  iuphar: null
  species_priority: null
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    cfg = load_pipeline_config(str(cfg_path))
    assert cfg.columns == DEFAULT_COLUMNS
    assert cfg.species_priority == ["Human", "Homo sapiens"]
    assert cfg.iuphar.approved_only is None
    assert cfg.iuphar.primary_target_only is True


def test_load_pipeline_config_rejects_invalid_types(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config_invalid.yaml"
    cfg_path.write_text(
        """
pipeline:
  retries: not-a-number
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="retries"):
        load_pipeline_config(str(cfg_path))


def test_load_pipeline_config_env_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = tmp_path / "config_env.yaml"
    cfg_path.write_text("pipeline:\n  list_format: json\n", encoding="utf-8")
    monkeypatch.setenv("CHEMBL_DA__PIPELINE__LIST_FORMAT", "pipe")
    monkeypatch.setenv("CHEMBL_DA__PIPELINE__IUPHAR__APPROVED_ONLY", "true")
    monkeypatch.setenv("CHEMBL_DA__PIPELINE__SPECIES_PRIORITY", '["Mouse", "Rat"]')
    cfg = load_pipeline_config(str(cfg_path))
    assert cfg.list_format == "pipe"
    assert cfg.iuphar.approved_only is True
    assert cfg.species_priority == ["Mouse", "Rat"]


def test_pipeline_single_target(monkeypatch):
    def chembl_fetch(ids, cfg=None):
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


def test_pipeline_fetches_isoforms_when_enabled(monkeypatch):
    def chembl_fetch(ids, cfg=None):
        return make_chembl_df(["P12345"])

    class RecordingUniProt(DummyUniProt):
        def fetch_isoforms_fasta(self, acc: str) -> List[str]:
            super().fetch_isoforms_fasta(acc)
            return [f">sp|{acc}-1|"]

    uni = RecordingUniProt(
        {
            "P12345": make_uniprot(
                "P12345",
                "Homo sapiens",
                9606,
                ["Eukaryota", "Chordata", "Mammalia", "Primates", "Hominidae"],
                hgnc="HGNC:1",
                include_isoform_comment=True,
            )
        }
    )
    hgnc = DummyHGNC({"P12345": HGNCRecord("P12345", "HGNC:1", "SYMB", "Name", "Prot")})
    gtop = DummyGtoP({})
    monkeypatch.setattr("pipeline_targets.resolve_target", fake_resolve)
    cfg = PipelineConfig(include_isoforms=True)

    cache: Dict[str, Dict[str, Any]] = {}
    df = run_pipeline(
        ["CHEMBL1"],
        cfg,
        chembl_fetcher=chembl_fetch,
        uniprot_client=uni,
        hgnc_client=hgnc,
        gtop_client=gtop,
        entry_cache=cache,
    )

    assert not df.empty
    assert uni.isoform_calls == ["P12345"]
    assert uni.batch_calls == [["P12345"]]
    assert cache["P12345"]["primaryAccession"] == "P12345"


def test_pipeline_batches_uniprot_requests(monkeypatch):
    def chembl_fetch(ids, cfg=None):
        return make_chembl_df(["P12345", "Q67890"])

    uni = DummyUniProt(
        {
            "P12345": make_uniprot(
                "P12345",
                "Homo sapiens",
                9606,
                ["Eukaryota", "Chordata", "Mammalia", "Primates", "Hominidae"],
            ),
            "Q67890": make_uniprot(
                "Q67890",
                "Homo sapiens",
                9606,
                ["Eukaryota", "Chordata", "Mammalia", "Primates", "Hominidae"],
            ),
        }
    )

    cfg = PipelineConfig(include_isoforms=True)
    df = run_pipeline(
        ["CHEMBL1"],
        cfg,
        chembl_fetcher=chembl_fetch,
        uniprot_client=uni,
        batch_size=10,
    )

    assert not df.empty
    assert uni.batch_calls == [["P12345", "Q67890"]]
    assert uni.entry_calls == []
    assert uni.isoform_calls == []



def test_run_pipeline_consumes_identifiers_lazily(monkeypatch):
    """Ensure pipeline iteration over identifiers is performed lazily."""

    state: dict[str, Any] = {
        "fetch_in_progress": False,
        "emitted": [],
        "consumed": [],
    }

    class SpyIds:
        def __init__(self, values: Iterable[str]) -> None:
            self._iterator = iter(values)

        def __iter__(self) -> "SpyIds":
            return self

        def __next__(self) -> str:
            try:
                value = next(self._iterator)
            except StopIteration:
                raise
            if not state["fetch_in_progress"]:
                raise AssertionError("Identifiers consumed eagerly")
            state["emitted"].append(value)
            return value

    target_to_uniprot = {
        "CHEMBL1": "P11111",
        "CHEMBL2": "P22222",
    }

    def chembl_fetch(ids_iter: Iterable[str], cfg: Any | None = None) -> pd.DataFrame:
        state["fetch_in_progress"] = True
        try:
            cleaned = list(ids_iter)
        finally:
            state["fetch_in_progress"] = False
        state["consumed"] = cleaned
        rows: list[dict[str, Any]] = []
        for index, chembl_id in enumerate(cleaned, start=1):
            accession = target_to_uniprot[chembl_id]
            rows.append(
                {
                    "target_chembl_id": chembl_id,
                    "pref_name": f"Target {chembl_id[-1]}",
                    "target_type": "SINGLE PROTEIN",
                    "organism": "Homo sapiens",
                    "target_components": json.dumps(
                        [
                            {
                                "component_id": index,
                                "accession": accession,
                                "component_type": "protein",
                                "component_description": "desc",
                            }
                        ],
                        sort_keys=True,
                    ),
                    "protein_classifications": json.dumps([], sort_keys=True),
                    "cross_references": json.dumps([], sort_keys=True),
                }
            )
        return pd.DataFrame(rows)

    records = {
        "P11111": make_uniprot(
            "P11111",
            "Homo sapiens",
            9606,
            ["Eukaryota", "Chordata", "Mammalia", "Primates", "Hominidae"],
        ),
        "P22222": make_uniprot(
            "P22222",
            "Homo sapiens",
            9606,
            ["Eukaryota", "Chordata", "Mammalia", "Primates", "Hominidae"],
        ),
    }

    spy_ids = SpyIds(["CHEMBL1", "CHEMBL1", "CHEMBL2"])
    uni = DummyUniProt(records)

    df = run_pipeline(
        spy_ids,
        PipelineConfig(include_isoforms=True),
        chembl_fetcher=chembl_fetch,
        uniprot_client=uni,
    )

    assert state["emitted"] == ["CHEMBL1", "CHEMBL1", "CHEMBL2"]
    assert state["consumed"] == ["CHEMBL1", "CHEMBL2"]
    assert not df.empty
    assert sorted(df["target_chembl_id"].unique()) == ["CHEMBL1", "CHEMBL2"]
    assert uni.batch_calls == [["P11111"], ["P22222"]]
    assert state["fetch_in_progress"] is False


def test_pipeline_selects_human_uniprot(monkeypatch):
    def chembl_fetch(ids, cfg=None):
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


def test_pipeline_missing_hgnc(monkeypatch):
    def chembl_fetch(ids, cfg=None):
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


def test_pipeline_reproducible(tmp_path, monkeypatch):
    def chembl_fetch(ids, cfg=None):
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


def test_pipeline_respects_config_columns(monkeypatch):
    """Pipeline uses column order defined in configuration."""

    def chembl_fetch(ids, cfg=None):
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


def test_add_protein_classification_network_error() -> None:
    df = pd.DataFrame({"uniprot_id_primary": ["P12345"]})

    def failing_fetch(_: Iterable[str]) -> Dict[str, dict]:
        raise requests.RequestException("boom")

    with pytest.raises(RuntimeError) as excinfo:
        add_protein_classification(df, failing_fetch)

    assert "P12345" in str(excinfo.value)
