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

    original_serialise = module.serialise_dataframe
    serialise_stats: dict[str, Any] = {"calls": 0}

    def tracking_serialise_dataframe(
        df: pd.DataFrame, list_format: str
    ) -> pd.DataFrame:
        serialise_stats["calls"] += 1
        serialise_stats["list_format"] = list_format
        serialise_stats["columns"] = list(df.columns)
        return original_serialise(df, list_format)

    monkeypatch.setattr(module, "serialise_dataframe", tracking_serialise_dataframe)

    original_write_metadata = module.write_cli_metadata
    metadata_stats: dict[str, Any] = {"calls": 0}

    def tracking_write_cli_metadata(*args: Any, **kwargs: Any) -> Path:
        metadata_stats["calls"] += 1
        metadata_stats["args"] = args
        metadata_stats["kwargs"] = kwargs
        return original_write_metadata(*args, **kwargs)

    monkeypatch.setattr(module, "write_cli_metadata", tracking_write_cli_metadata)

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
                "synonyms_all": [
                    ["z-last"],
                    ["a-first", "b-first"],
                    ["m-middle"],
                ],
                "cross_references": [
                    {"source": "last"},
                    {"source": "first"},
                    {"source": "middle"},
                ],
            }
        )
        for column in module.IUPHAR_CLASS_COLUMNS:
            data[column] = ""
        return data

    def fake_build_clients(*_args: Any, **_kwargs: Any) -> tuple[Any, ...]:
        return DummyUniClient(), object(), object(), None, None, []

    captured: dict[str, Any] = {"call_count": 0}

    def fake_analyze_table_quality(table: pd.DataFrame, table_name: str) -> None:
        captured["call_count"] = captured.get("call_count", 0) + 1
        captured["table"] = table.copy()
        captured["table_name"] = table_name

    monkeypatch.setattr(module, "load_pipeline_config", fake_load_pipeline_config)
    monkeypatch.setattr(module, "fetch_targets", fake_fetch_targets)
    monkeypatch.setattr(module, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(module, "build_clients", fake_build_clients)

    enrich_call: dict[str, Any] = {}

    def fake_enrich_client(*args: Any, **kwargs: Any) -> DummyEnrichClient:
        enrich_call["args"] = args
        enrich_call["kwargs"] = kwargs
        return DummyEnrichClient()

    monkeypatch.setattr(module, "UniProtEnrichClient", fake_enrich_client)
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
    assert result_df["synonyms_all"].tolist() == [
        '["a-first","b-first"]',
        '["m-middle"]',
        '["z-last"]',
    ]
    assert result_df["cross_references"].tolist() == [
        '{"source": "first"}',
        '{"source": "middle"}',
        '{"source": "last"}',
    ]

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
    assert captured["call_count"] == 1
    assert serialise_stats["calls"] == 1
    assert serialise_stats["list_format"] == "json"
    assert metadata_stats["calls"] == 1
    assert metadata_stats["kwargs"].get("meta_path") is None
    assert enrich_call["kwargs"].get("cache_config") is None


def _identity_frame(df: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
    """Return ``df`` unchanged for monkeypatched processing steps."""

    return df


def test_pipeline_targets_cli_filters_invalid_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module: Any = importlib.import_module("scripts.pipeline_targets_main")

    input_csv = tmp_path / "input.csv"
    input_csv.write_text(
        "target_chembl_id\nCHEMBL10\nnan\nCHEMBL11\nNone\n\nCHEMBL10\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    captured: dict[str, Any] = {}

    def fake_fetch_targets(ids: list[str], cfg: Any, batch_size: int) -> pd.DataFrame:
        captured["ids"] = list(ids)
        return pd.DataFrame({"target_chembl_id": ids})

    def fake_run_pipeline(ids: list[str], *_args: Any, **_kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "target_chembl_id": ids,
                "uniprot_id_primary": [f"P{i}" for i in range(len(ids))],
                "gene_symbol": [f"GENE{i}" for i in range(len(ids))],
                "hgnc_id": [f"HGNC:{i}" for i in range(len(ids))],
            }
        )

    class DummyEnrichClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - simple
            pass

        def fetch_all(self, accessions: list[str]) -> dict[str, dict[str, str]]:  # pragma: no cover - simple
            return {acc: {} for acc in accessions}

    def fake_build_clients(*_args: Any, **_kwargs: Any) -> tuple[Any, ...]:
        return object(), object(), object(), None, None, []

    monkeypatch.setattr(module, "fetch_targets", fake_fetch_targets)
    monkeypatch.setattr(module, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(module, "add_uniprot_fields", _identity_frame)
    monkeypatch.setattr(module, "merge_chembl_fields", lambda df, _: df)
    monkeypatch.setattr(module, "add_activity_fields", _identity_frame)
    monkeypatch.setattr(module, "add_isoform_fields", _identity_frame)
    def _ensure_iuphar_columns(df: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        frame = df.copy()
        for column in module.IUPHAR_CLASS_COLUMNS:
            if column not in frame.columns:
                frame[column] = ""
        return frame

    monkeypatch.setattr(module, "add_iuphar_classification", _ensure_iuphar_columns)
    monkeypatch.setattr(module, "add_protein_classification", _identity_frame)
    monkeypatch.setattr(module, "UniProtEnrichClient", lambda *a, **k: DummyEnrichClient())
    monkeypatch.setattr(module, "build_clients", fake_build_clients)
    monkeypatch.setattr(module, "analyze_table_quality", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "write_cli_metadata",
        lambda *args, **kwargs: tmp_path / "meta.yaml",
    )

    argv = [
        "pipeline_targets_main",
        "--input",
        str(input_csv),
        "--output",
        str(tmp_path / "out.csv"),
        "--config",
        str(config_path),
        "--iuphar-target",
        str(Path("tests/data/iuphar_target.csv")),
        "--iuphar-family",
        str(Path("tests/data/iuphar_family.csv")),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    module.main()

    assert captured.get("ids") == ["CHEMBL10", "CHEMBL11"]


def test_pipeline_targets_cli_rejects_invalid_batch_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module: Any = importlib.import_module("scripts.pipeline_targets_main")

    input_csv = tmp_path / "input.csv"
    input_csv.write_text("target_chembl_id\nCHEMBL1\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    argv = [
        "pipeline_targets_main",
        "--input",
        str(input_csv),
        "--output",
        str(tmp_path / "out.csv"),
        "--config",
        str(config_path),
        "--batch-size",
        "0",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit):
        module.main()


def _dummy_enrich_client(*_args: Any, **_kwargs: Any) -> Any:
    """Return a dummy enrich client compatible with the pipeline contract."""

    class _Dummy:
        def fetch_all(self, _accessions: list[str]) -> dict[str, dict[str, Any]]:
            return {}

    return _Dummy()


def _noop_analyze_table_quality(*_args: Any, **_kwargs: Any) -> None:
    """No-op replacement for :func:`analyze_table_quality` in tests."""

    return None


def _fake_write_metadata(*_args: Any, **_kwargs: Any) -> Path:
    """Return a placeholder metadata path for CLI unit tests."""

    return Path("meta.yaml")


def test_pipeline_targets_cli_uses_configured_list_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pipeline should honour YAML list_format when CLI flag is absent."""

    module: Any = importlib.import_module("scripts.pipeline_targets_main")

    serialise_stats: dict[str, Any] = {}

    def fake_serialise_dataframe(df: pd.DataFrame, list_format: str) -> pd.DataFrame:
        serialise_stats["list_format"] = list_format
        return df

    monkeypatch.setattr(module, "serialise_dataframe", fake_serialise_dataframe)

    input_csv = tmp_path / "input.csv"
    input_csv.write_text("target_chembl_id\nCHEMBL1\n", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text("pipeline:\n  list_format: pipe\n", encoding="utf-8")

    class DummyUniClient:
        def fetch_entry_json(self, accession: str) -> dict[str, Any]:
            return {}

        def fetch_entries_json(
            self, accessions: list[str], batch_size: int = 0
        ) -> dict[str, dict[str, Any]]:
            return {accession: {} for accession in accessions}

    def fake_build_clients(*_args: Any, **_kwargs: Any) -> tuple[Any, ...]:
        return DummyUniClient(), object(), object(), None, None, []

    monkeypatch.setattr(module, "build_clients", fake_build_clients)

    def fake_fetch_targets(ids: list[str], *_args: Any, **_kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame({"target_chembl_id": ids})

    monkeypatch.setattr(module, "fetch_targets", fake_fetch_targets)

    def fake_run_pipeline(ids: list[str], *_args: Any, **_kwargs: Any) -> pd.DataFrame:
        data: dict[str, Any] = {
            "target_chembl_id": ids,
            "uniprot_id_primary": ["P001" for _ in ids],
            "gene_symbol": ["GENE" for _ in ids],
        }
        for column in module.IUPHAR_CLASS_COLUMNS:
            data[column] = [""] * len(ids)
        return pd.DataFrame(data)

    monkeypatch.setattr(module, "run_pipeline", fake_run_pipeline)

    monkeypatch.setattr(module, "add_uniprot_fields", _identity_frame)
    monkeypatch.setattr(module, "merge_chembl_fields", _identity_frame)
    monkeypatch.setattr(module, "add_activity_fields", _identity_frame)
    monkeypatch.setattr(module, "add_protein_classification", _identity_frame)
    monkeypatch.setattr(module, "UniProtEnrichClient", _dummy_enrich_client)
    monkeypatch.setattr(module, "analyze_table_quality", _noop_analyze_table_quality)
    monkeypatch.setattr(
        module, "write_cli_metadata", lambda *args, **kwargs: tmp_path / "meta.yaml"
    )

    argv = [
        "pipeline_targets_main",
        "--input",
        str(input_csv),
        "--output",
        str(tmp_path / "targets.csv"),
        "--config",
        str(config_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    module.main()

    assert serialise_stats["list_format"] == "pipe"


def test_pipeline_targets_cli_respects_yaml_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure YAML configuration is honoured when overriding flags are absent."""

    module: Any = importlib.import_module("scripts.pipeline_targets_main")

    input_csv = tmp_path / "input.csv"
    input_csv.write_text("target_chembl_id\nCHEMBL1\n", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
pipeline:
  list_format: pipe
  species_priority:
    - "Homo sapiens"
    - "Mus musculus"
  iuphar:
    approved_only: true
    primary_target_only: false
orthologs:
  enabled: true
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    class DummyUniClient:
        def fetch_entry_json(self, accession: str) -> dict[str, Any]:
            return {}

        def fetch_entries_json(
            self, accessions: list[str], batch_size: int = 0
        ) -> dict[str, dict[str, Any]]:
            return {accession: {} for accession in accessions}

    def fake_build_clients(*_args: Any, **kwargs: Any) -> tuple[Any, ...]:
        captured["with_orthologs"] = kwargs.get("with_orthologs")
        return DummyUniClient(), object(), object(), None, None, []

    monkeypatch.setattr(module, "build_clients", fake_build_clients)

    def fake_fetch_targets(ids: list[str], *_args: Any, **_kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame({"target_chembl_id": ids})

    monkeypatch.setattr(module, "fetch_targets", fake_fetch_targets)

    def fake_run_pipeline(
        ids: list[str], pipeline_cfg: Any, *_args: Any, **_kwargs: Any
    ) -> pd.DataFrame:
        captured["list_format"] = pipeline_cfg.list_format
        captured["species_priority"] = list(pipeline_cfg.species_priority)
        captured["approved_only"] = pipeline_cfg.iuphar.approved_only
        captured["primary_target_only"] = pipeline_cfg.iuphar.primary_target_only
        data: dict[str, Any] = {
            "target_chembl_id": ids,
            "uniprot_id_primary": ["P001" for _ in ids],
            "gene_symbol": ["GENE" for _ in ids],
        }
        for column in module.IUPHAR_CLASS_COLUMNS:
            data[column] = [""] * len(ids)
        return pd.DataFrame(data)

    monkeypatch.setattr(module, "run_pipeline", fake_run_pipeline)

    monkeypatch.setattr(module, "add_uniprot_fields", _identity_frame)
    monkeypatch.setattr(module, "merge_chembl_fields", _identity_frame)
    monkeypatch.setattr(module, "add_activity_fields", _identity_frame)
    monkeypatch.setattr(module, "add_protein_classification", _identity_frame)
    monkeypatch.setattr(module, "UniProtEnrichClient", _dummy_enrich_client)
    monkeypatch.setattr(module, "analyze_table_quality", _noop_analyze_table_quality)
    monkeypatch.setattr(
        module, "write_cli_metadata", lambda *args, **kwargs: tmp_path / "meta.yaml"
    )

    argv = [
        "pipeline_targets_main",
        "--input",
        str(input_csv),
        "--output",
        str(tmp_path / "targets.csv"),
        "--config",
        str(config_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    module.main()

    assert captured["list_format"] == "pipe"
    assert captured["species_priority"] == ["Homo sapiens", "Mus musculus"]
    assert captured["approved_only"] is True
    assert captured["primary_target_only"] is False
    assert captured["with_orthologs"] is True


def test_pipeline_targets_cli_overrides_specific_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI flags should update only the targeted configuration fields."""

    module: Any = importlib.import_module("scripts.pipeline_targets_main")

    input_csv = tmp_path / "input.csv"
    input_csv.write_text("target_chembl_id\nCHEMBL1\n", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
pipeline:
  list_format: pipe
  species_priority:
    - "Homo sapiens"
    - "Mus musculus"
  iuphar:
    approved_only: true
    primary_target_only: false
orthologs:
  enabled: false
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    class DummyUniClient:
        def fetch_entry_json(self, accession: str) -> dict[str, Any]:
            return {}

        def fetch_entries_json(
            self, accessions: list[str], batch_size: int = 0
        ) -> dict[str, dict[str, Any]]:
            return {accession: {} for accession in accessions}

    def fake_build_clients(*_args: Any, **kwargs: Any) -> tuple[Any, ...]:
        captured["with_orthologs"] = kwargs.get("with_orthologs")
        return DummyUniClient(), object(), object(), None, None, []

    monkeypatch.setattr(module, "build_clients", fake_build_clients)

    def fake_fetch_targets(ids: list[str], *_args: Any, **_kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame({"target_chembl_id": ids})

    monkeypatch.setattr(module, "fetch_targets", fake_fetch_targets)

    def fake_run_pipeline(
        ids: list[str], pipeline_cfg: Any, *_args: Any, **_kwargs: Any
    ) -> pd.DataFrame:
        captured["list_format"] = pipeline_cfg.list_format
        captured["species_priority"] = list(pipeline_cfg.species_priority)
        captured["approved_only"] = pipeline_cfg.iuphar.approved_only
        captured["primary_target_only"] = pipeline_cfg.iuphar.primary_target_only
        data: dict[str, Any] = {
            "target_chembl_id": ids,
            "uniprot_id_primary": ["P001" for _ in ids],
            "gene_symbol": ["GENE" for _ in ids],
        }
        for column in module.IUPHAR_CLASS_COLUMNS:
            data[column] = [""] * len(ids)
        return pd.DataFrame(data)

    monkeypatch.setattr(module, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(module, "add_uniprot_fields", _identity_frame)
    monkeypatch.setattr(module, "merge_chembl_fields", _identity_frame)
    monkeypatch.setattr(module, "add_activity_fields", _identity_frame)
    monkeypatch.setattr(module, "add_protein_classification", _identity_frame)
    monkeypatch.setattr(module, "UniProtEnrichClient", _dummy_enrich_client)
    monkeypatch.setattr(module, "analyze_table_quality", _noop_analyze_table_quality)
    monkeypatch.setattr(module, "write_cli_metadata", _fake_write_metadata)

    argv = [
        "pipeline_targets_main",
        "--input",
        str(input_csv),
        "--output",
        str(tmp_path / "targets.csv"),
        "--config",
        str(config_path),
        "--list-format",
        "json",
        "--species",
        "Canis familiaris, Homo sapiens",
        "--approved-only",
        "false",
        "--primary-target-only",
        "true",
        "--with-orthologs",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    module.main()

    assert captured["list_format"] == "json"
    assert captured["species_priority"] == [
        "Canis familiaris",
        "Homo sapiens",
        "Mus musculus",
    ]
    assert captured["approved_only"] is False
    assert captured["primary_target_only"] is True
    assert captured["with_orthologs"] is True


def test_pipeline_targets_cli_accepts_null_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI should tolerate optional sections set to ``null`` in YAML."""

    module: Any = importlib.import_module("scripts.pipeline_targets_main")

    input_csv = tmp_path / "input.csv"
    input_csv.write_text("target_chembl_id\nCHEMBL1\n", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
pipeline:
  columns: null
  iuphar: null
orthologs: null
chembl: null
uniprot_enrich: null
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    class DummyUniClient:
        def fetch_entry_json(self, accession: str) -> dict[str, Any]:
            return {}

        def fetch_entries_json(
            self, accessions: list[str], batch_size: int = 0
        ) -> dict[str, dict[str, Any]]:
            return {accession: {} for accession in accessions}

    def fake_build_clients(*_args: Any, **kwargs: Any) -> tuple[Any, ...]:
        captured["with_orthologs"] = kwargs.get("with_orthologs")
        captured["default_cache"] = kwargs.get("default_cache")
        return DummyUniClient(), object(), object(), None, None, []

    monkeypatch.setattr(module, "build_clients", fake_build_clients)

    def fake_fetch_targets(ids: list[str], cfg: Any, batch_size: int) -> pd.DataFrame:
        captured["chembl_columns"] = list(cfg.columns)
        return pd.DataFrame({"target_chembl_id": ids})

    monkeypatch.setattr(module, "fetch_targets", fake_fetch_targets)

    def fake_run_pipeline(ids: list[str], *_args: Any, **_kwargs: Any) -> pd.DataFrame:
        data: dict[str, Any] = {
            "target_chembl_id": ids,
            "uniprot_id_primary": ["P001" for _ in ids],
            "gene_symbol": ["GENE" for _ in ids],
        }
        for column in module.IUPHAR_CLASS_COLUMNS:
            data[column] = [""] * len(ids)
        return pd.DataFrame(data)

    monkeypatch.setattr(module, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(module, "add_uniprot_fields", _identity_frame)
    monkeypatch.setattr(module, "merge_chembl_fields", _identity_frame)
    monkeypatch.setattr(module, "add_activity_fields", _identity_frame)
    monkeypatch.setattr(module, "add_protein_classification", _identity_frame)
    monkeypatch.setattr(module, "UniProtEnrichClient", _dummy_enrich_client)
    monkeypatch.setattr(module, "analyze_table_quality", _noop_analyze_table_quality)
    monkeypatch.setattr(module, "write_cli_metadata", _fake_write_metadata)

    argv = [
        "pipeline_targets_main",
        "--input",
        str(input_csv),
        "--output",
        str(tmp_path / "targets.csv"),
        "--config",
        str(config_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    module.main()

    assert captured["with_orthologs"] is False
    assert "target_chembl_id" in captured["chembl_columns"]
