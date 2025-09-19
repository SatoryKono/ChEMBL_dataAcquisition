from __future__ import annotations

import importlib
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "get_uniprot_target_data.py"


def test_missing_column_exits_cleanly(tmp_path: Path) -> None:
    """The UniProt target CLI should exit gracefully when the column is absent."""

    input_csv = tmp_path / "input.csv"
    input_csv.write_text("identifier\nP12345\n", encoding="utf-8")
    output_csv = tmp_path / "results.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(input_csv),
            "--output",
            str(output_csv),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "does not contain the required" in result.stderr
    assert "Traceback" not in result.stderr

    meta_path = output_csv.with_name(f"{output_csv.name}.meta.yaml")
    assert meta_path.exists()
    metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "error"
    assert "does not contain the required" in metadata["error"]
    assert metadata["sha256"] is None


def test_missing_input_file_exits_with_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The CLI should report when the input file cannot be located."""

    import scripts.get_uniprot_target_data as cli

    real_path = cli.Path

    class MissingPathStub:
        """Stand-in path reporting that the referenced file is absent."""

        def __init__(self, raw: str):
            self._raw = raw

        def exists(self) -> bool:
            return False

        def is_file(self) -> bool:
            return False

        def __str__(self) -> str:
            return self._raw

        def __fspath__(self) -> str:
            return self._raw

    def fake_path(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if args and args[0] == "missing.csv" and len(args) == 1:
            return MissingPathStub("missing.csv")
        return real_path(*args, **kwargs)

    monkeypatch.setattr(cli, "Path", fake_path)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--input", "missing.csv", "--output", str(tmp_path / "output.csv")])

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Input file missing.csv does not exist" in captured.err


def test_cli_uses_custom_batch_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLI should respect ``--batch-size`` overrides when batching accessions."""

    module = importlib.import_module("scripts.get_uniprot_target_data")

    custom_batch_size = 3
    accessions = [f"P{index:05d}" for index in range(custom_batch_size * 2 + 1)]

    input_csv = tmp_path / "input.csv"
    input_csv.write_text(
        "uniprot_id\n" + "\n".join(accessions) + "\n",
        encoding="utf-8",
    )
    output_csv = tmp_path / "output.csv"

    class DummyOutputConfig:
        list_format = "json"
        include_sequence = False
        sep = ","
        encoding = "utf-8"

    class DummyUniProtConfig:
        def __init__(self) -> None:
            self.include_isoforms = False
            self.use_fasta_stream_for_isoform_ids = False
            self.cache = None
            self.base_url = "https://example.uniprot"
            self.timeout_sec = 1.0
            self.retries = 0
            self.rps = 10.0
            self.batch_size = module.DEFAULT_BATCH_SIZE
            self.fields: list[str] = []
            self.columns: list[str] = []

        def model_dump(self) -> dict[str, list[str]]:
            return {"fields": self.fields, "columns": self.columns}

    class DummyOrthologsConfig:
        enabled = False

    class DummyConfig:
        def __init__(self) -> None:
            self.http_cache = None
            self.output = DummyOutputConfig()
            self.uniprot = DummyUniProtConfig()
            self.orthologs = DummyOrthologsConfig()

    config = DummyConfig()

    monkeypatch.setattr(module.yaml, "safe_load", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        module,
        "load_uniprot_target_config",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(
        "library.logging_utils.configure_logging", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr("library.cli_common.ensure_output_dir", lambda path: Path(path))
    monkeypatch.setattr(
        "library.cli_common.serialise_dataframe",
        lambda df, list_format, *, inplace=False: df,
    )
    monkeypatch.setattr(
        "library.cli_common.write_cli_metadata", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "library.cli_common.analyze_table_quality", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "library.cli_common.resolve_cli_sidecar_paths",
        lambda _path: (
            tmp_path / "meta.yaml",
            tmp_path / "errors.json",
            tmp_path / "quality",
        ),
    )
    monkeypatch.setattr("library.io_utils.write_rows", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "library.uniprot_normalize.extract_ensembl_gene_ids",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "library.uniprot_normalize.extract_isoforms", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        "library.uniprot_normalize.normalize_entry",
        lambda data, include_seq, isoforms: {
            "uniprot_id": data["primaryAccession"],
        },
    )
    monkeypatch.setattr(
        "library.uniprot_normalize.output_columns",
        lambda *_args, **_kwargs: ["uniprot_id"],
    )
    monkeypatch.setattr(
        "library.http_client.CacheConfig.from_dict", lambda *_args, **_kwargs: None
    )

    fetch_calls: list[tuple[tuple[str, ...], int]] = []

    class RecordingClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def fetch_entries_json(
            self, accessions: list[str], *, batch_size: int = 0
        ) -> dict[str, dict[str, str]]:
            fetch_calls.append((tuple(accessions), batch_size))
            return {acc: {"primaryAccession": acc} for acc in accessions}

        def fetch_isoforms_fasta(self, *_args: Any, **_kwargs: Any) -> list[str]:
            raise AssertionError("Isoform fetching should be disabled in this test")

        def fetch_entry_json(self, *_args: Any, **_kwargs: Any) -> dict[str, str]:
            raise AssertionError("Single-entry fetches should not be used")

    monkeypatch.setattr("library.uniprot_client.UniProtClient", RecordingClient)

    module.main(
        [
            "--input",
            str(input_csv),
            "--output",
            str(output_csv),
            "--batch-size",
            str(custom_batch_size),
        ]
    )

    assert fetch_calls
    expected_batches = math.ceil(len(accessions) / custom_batch_size)
    assert len(fetch_calls) == expected_batches
    for batch_accessions, call_batch_size in fetch_calls:
        assert call_batch_size == custom_batch_size
        assert len(batch_accessions) <= custom_batch_size
    assert sum(len(batch) for batch, _size in fetch_calls) == len(accessions)


def test_cli_warns_when_ortholog_flags_disagree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CLI should emit and persist warnings when overriding ortholog settings."""

    module = importlib.import_module("scripts.get_uniprot_target_data")

    input_csv = tmp_path / "input.csv"
    input_csv.write_text("uniprot_id\nP12345\n", encoding="utf-8")
    output_csv = tmp_path / "output.csv"

    class DummyOutputConfig:
        list_format = "json"
        include_sequence = False
        sep = ","
        encoding = "utf-8"

    class DummyUniProtConfig:
        def __init__(self) -> None:
            self.include_isoforms = False
            self.use_fasta_stream_for_isoform_ids = False
            self.cache = None
            self.base_url = "https://example.uniprot"
            self.timeout_sec = 1.0
            self.retries = 0
            self.rps = 10.0
            self.batch_size = module.DEFAULT_BATCH_SIZE
            self.fields: list[str] = []
            self.columns: list[str] = []

        def model_dump(self) -> dict[str, list[str]]:
            return {"fields": self.fields, "columns": self.columns}

    class DummyOrthologsConfig:
        def __init__(self) -> None:
            self.enabled = False
            self.cache = None
            self.timeout_sec = 1.0
            self.retries = 0
            self.backoff_base_sec = 0.5
            self.rate_limit_rps = 2.0
            self.target_species = ["human"]

    class DummyConfig:
        def __init__(self) -> None:
            self.http_cache = None
            self.output = DummyOutputConfig()
            self.uniprot = DummyUniProtConfig()
            self.orthologs = DummyOrthologsConfig()

    config = DummyConfig()

    real_yaml_safe_load = module.yaml.safe_load
    monkeypatch.setattr(
        module.yaml,
        "safe_load",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        module,
        "load_uniprot_target_config",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(
        module,
        "_resolve_uniprot_fields",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "library.logging_utils.configure_logging", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "library.http_client.CacheConfig.from_dict", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr("library.io_utils.write_rows", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "library.cli_common.serialise_dataframe",
        lambda df, list_format, *, inplace=False: df,
    )
    monkeypatch.setattr(
        "library.cli_common.analyze_table_quality",
        lambda *_args, **_kwargs: None,
    )

    def fake_read_ids(*_args: Any, **_kwargs: Any) -> Iterator[str]:
        yield "P12345"

    monkeypatch.setattr("library.io.read_ids", fake_read_ids)

    class DummyUniProtClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def fetch_entries_json(
            self, accessions: list[str], *, batch_size: int = 0
        ) -> dict[str, dict[str, str]]:
            return {
                accession: {
                    "primaryAccession": accession,
                    "organism": {"scientificName": "Homo sapiens"},
                }
                for accession in accessions
            }

        def fetch_isoforms_fasta(self, *_args: Any, **_kwargs: Any) -> list[str]:
            raise AssertionError("Isoform retrieval should be disabled")

        def fetch_entry_json(self, *_args: Any, **_kwargs: Any) -> dict[str, str]:
            return {}

    class DummyEnsemblClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def get_orthologs(self, _gene_id: str, _species: list[str]) -> list[Any]:
            return []

    class DummyOmaClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def get_orthologs_by_uniprot(self, *_args: Any, **_kwargs: Any) -> list[Any]:
            return []

    monkeypatch.setattr("library.uniprot_client.UniProtClient", DummyUniProtClient)
    monkeypatch.setattr("library.orthologs.EnsemblHomologyClient", DummyEnsemblClient)
    monkeypatch.setattr("library.orthologs.OmaClient", DummyOmaClient)
    monkeypatch.setattr(
        "library.uniprot_normalize.extract_ensembl_gene_ids",
        lambda *_args, **_kwargs: ["ENSG000001"],
    )
    monkeypatch.setattr(
        "library.uniprot_normalize.extract_isoforms",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "library.uniprot_normalize.normalize_entry",
        lambda data, include_seq, isoforms: {
            "uniprot_id": data["primaryAccession"],
            "organism_name": "Homo sapiens",
        },
    )
    monkeypatch.setattr(
        "library.uniprot_normalize.output_columns",
        lambda *_args, **_kwargs: ["uniprot_id", "organism_name"],
    )

    caplog.set_level("WARNING", logger=module.LOGGER.name)

    module.main(
        [
            "--input",
            str(input_csv),
            "--output",
            str(output_csv),
            "--with-orthologs",
        ]
    )

    expected_message = "Ortholog enrichment enabled via CLI override although configuration disables it."
    warning_messages = [record.message for record in caplog.records]
    assert expected_message in warning_messages

    meta_path = output_csv.with_name(f"{output_csv.name}.meta.yaml")
    payload = real_yaml_safe_load(meta_path.read_text(encoding="utf-8"))
    assert payload["warnings"] == [expected_message]
