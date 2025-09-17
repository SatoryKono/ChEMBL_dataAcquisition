from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from library.io_utils import serialise_cell
from pipeline_targets import (
    PipelineConfig,
    _load_serialised_list,
    run_pipeline,
)


def _pipe_encode(items: List[Any]) -> str:
    """Return a pipe-serialised string compatible with the pipeline helpers."""

    return str(serialise_cell(items, "pipe"))


def test_load_serialised_list_pipe_json_roundtrip() -> None:
    values = [{"a": 1}, {"b": 2}]
    serialised = _pipe_encode(values)
    assert _load_serialised_list(serialised, "pipe") == values


def test_load_serialised_list_pipe_plain_strings() -> None:
    serialised = "alpha|beta|gamma"
    assert _load_serialised_list(serialised, "pipe") == ["alpha", "beta", "gamma"]


def test_load_serialised_list_pipe_with_escaped_pipes() -> None:
    serialised = _pipe_encode(["alpha|beta", {"name": "value|with|pipes"}])
    assert _load_serialised_list(serialised, "pipe") == [
        "alpha|beta",
        {"name": "value|with|pipes"},
    ]


class _DummyUniProt:
    def __init__(self, records: Dict[str, Dict[str, Any]]):
        self.records = records

    def fetch(self, accession: str) -> Dict[str, Any] | None:
        return self.records.get(accession)


def _make_pipe_chembl_df(accessions: List[str]) -> pd.DataFrame:
    comps = [
        {
            "component_id": idx + 1,
            "accession": accession,
            "component_type": "protein",
            "component_description": "desc",
        }
        for idx, accession in enumerate(accessions)
    ]
    return pd.DataFrame(
        [
            {
                "target_chembl_id": "CHEMBL_PIPE",
                "pref_name": "Pipe Target",
                "target_type": "SINGLE PROTEIN",
                "organism": "Homo sapiens",
                "target_components": _pipe_encode(comps),
                "protein_classifications": _pipe_encode(["ClassA"]),
                "cross_references": _pipe_encode(
                    [
                        {"xref_db": "ChEMBL", "xref_id": "CHEMBL_PIPE"},
                        {"xref_db": "Other", "xref_id": "O1"},
                    ]
                ),
                "protein_synonym_list": _pipe_encode(["SynPipe"]),
                "gene_symbol_list": _pipe_encode(["PIPE1"]),
            }
        ]
    )


def test_run_pipeline_decodes_pipe_serialised_lists(monkeypatch) -> None:
    def chembl_fetch(ids, cfg=None):
        return _make_pipe_chembl_df(["P11111", "P22222"])

    def fake_normalize_entry(raw, *, include_sequence=False, isoforms=None):
        accession = raw["primaryAccession"]
        return {
            "uniprot_id": accession,
            "organism_name": "Homo sapiens",
            "taxon_id": 9606,
            "lineage_superkingdom": "Eukaryota",
            "lineage_phylum": "Chordata",
            "lineage_class": "Mammalia",
            "lineage_order": "Primates",
            "lineage_family": "Hominidae",
            "gene_primary": f"GENE{accession}",
            "gene_synonyms": [f"{accession}_SYN"],
            "protein_recommended_name": f"Protein {accession}",
            "protein_alternative_names": [f"Alt {accession}"],
            "isoform_ids_all": [],
            "isoform_names": [],
        }

    monkeypatch.setattr("pipeline_targets.normalize_entry", fake_normalize_entry)

    uniprot_records = {
        "P11111": {"primaryAccession": "P11111"},
        "P22222": {"primaryAccession": "P22222"},
    }
    uni = _DummyUniProt(uniprot_records)

    cfg = PipelineConfig(list_format="pipe")
    df = run_pipeline(
        ["CHEMBL_PIPE"],
        cfg,
        chembl_fetcher=chembl_fetch,
        uniprot_client=uni,
    )

    row = df.iloc[0]
    assert row["uniprot_id_primary"] == "P11111"
    assert row["uniprot_ids_all"] == "P11111|P22222"
    assert row["protein_class_L1"] == "ClassA"
    synonyms_all = set(row["synonyms_all"].split("|"))
    assert {"SynPipe", "P11111_SYN"}.issubset(synonyms_all)
