"""Integration tests for the pipeline targets CLI entry point."""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_pipeline_targets_cli_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the CLI creates the CSV and companion metadata file."""

    module: Any = importlib.import_module("scripts.pipeline_targets_main")

    input_csv = tmp_path / "input.csv"
    input_csv.write_text(
        "target_chembl_id\nCHEMBL2\nCHEMBL1\nCHEMBL2\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    class DummyUniClient:
        def fetch_entry_json(self, accession: str) -> dict[str, Any]:
            return {}

        def fetch_entries_json(
            self, accessions: list[str], batch_size: int = 0
        ) -> dict[str, dict[str, Any]]:
            return {accession: {} for accession in accessions}

    class DummyEnrichClient:
        def fetch_all(self, accessions: list[str]) -> dict[str, dict[str, str]]:
            return {accession: {} for accession in accessions}

    def fake_load_pipeline_config(_: str) -> Any:
        cfg = module.PipelineConfig()
        return cfg

    def fake_fetch_targets(ids: list[str], _: Any, batch_size: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "target_chembl_id": ids,
                "pref_name": [f"pref_{identifier}" for identifier in ids],
            }
        )

    def fake_run_pipeline(ids: list[str], *_args: Any, **kwargs: Any) -> pd.DataFrame:
        progress_callback = kwargs.get("progress_callback")
        if callable(progress_callback):
            for _ in ids:
                progress_callback(1)
        data = pd.DataFrame(
            {
                "target_chembl_id": ["CHEMBL2", "CHEMBL1", "CHEMBL2"],
                "uniprot_id_primary": ["P200", "P100", "P010"],
                "gene_symbol": ["BBB", "AAA", "AAC"],
            }
        )
        for column in module.IUPHAR_CLASS_COLUMNS:
            data[column] = ""
        return data

    def fake_build_clients(*_args: Any, **_kwargs: Any) -> tuple[Any, ...]:
        return DummyUniClient(), object(), object(), None, None, []

    captured: dict[str, Any] = {}

    def fake_analyze_table_quality(table: pd.DataFrame, table_name: str) -> None:
        captured["table"] = table.copy()
        captured["table_name"] = table_name

    monkeypatch.setattr(module, "load_pipeline_config", fake_load_pipeline_config)
    monkeypatch.setattr(module, "fetch_targets", fake_fetch_targets)
    monkeypatch.setattr(module, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(module, "build_clients", fake_build_clients)
    monkeypatch.setattr(module, "UniProtEnrichClient", lambda: DummyEnrichClient())
    monkeypatch.setattr(module, "analyze_table_quality", fake_analyze_table_quality)

    argv = [
        "pipeline_targets_main",
        "--input",
        str(input_csv),
        "--output",
        str(tmp_path / "nested" / "targets.csv"),
        "--config",
        str(config_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    output_path = Path(argv[4])
    assert not output_path.parent.exists()

    module.main()

    assert output_path.exists()
    assert output_path.parent.exists()

    result_df = pd.read_csv(output_path)
    assert list(result_df["target_chembl_id"]) == ["CHEMBL1", "CHEMBL2", "CHEMBL2"]
    assert list(result_df["uniprot_id_primary"]) == ["P100", "P010", "P200"]

    meta_path = output_path.with_suffix(f"{output_path.suffix}.meta.yaml")
    assert meta_path.exists()

    metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    assert metadata["output"] == str(output_path)
    assert metadata["rows"] == 3
    assert metadata["columns"] == len(result_df.columns)
    assert metadata["config"]["input"] == str(input_csv)

    expected_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    assert metadata["sha256"] == expected_hash

    assert captured["table_name"] == str(output_path.with_suffix(""))
    assert list(captured["table"]["target_chembl_id"]) == [
        "CHEMBL1",
        "CHEMBL2",
        "CHEMBL2",
    ]
