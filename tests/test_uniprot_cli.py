from __future__ import annotations

import importlib
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pytest

from uniprot_enrich import enrich_uniprot

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "uniprot_enrich_main.py"
DATA_FILE = Path(__file__).parent / "data" / "uniprot_sample.csv"


class DummyOutputConfig:
    """Minimal output configuration used when stubbing UniProt CLI config."""

    list_format = "json"
    include_sequence = False
    sep = ","
    encoding = "utf-8"


class DummyUniProtConfig:
    """Minimal UniProt API configuration used for CLI tests."""

    def __init__(self) -> None:
        self.include_isoforms = False
        self.use_fasta_stream_for_isoform_ids = True
        self.cache = None
        self.base_url = "https://example.uniprot"
        self.timeout_sec = 1.0
        self.retries = 0
        self.rps = 10.0
        self.fields: list[str] = []
        self.columns: list[str] = []

    def model_dump(self) -> dict[str, list[str]]:
        """Return field/column metadata to satisfy client construction."""

        return {"fields": self.fields, "columns": self.columns}


class DummyOrthologsConfig:
    """Ortholog configuration stub disabling enrichment by default."""

    enabled = False


class DummyConfig:
    """Container aggregating stubbed configuration sections for the CLI."""

    def __init__(self) -> None:
        self.http_cache = None
        self.output = DummyOutputConfig()
        self.uniprot = DummyUniProtConfig()
        self.orthologs = DummyOrthologsConfig()


def _monkeypatch_uniprot_cli(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    accessions: Sequence[str],
    *,
    patch_yaml: bool,
) -> tuple[list[str], list[Any]]:
    """Replace side-effect heavy CLI dependencies with lightweight stubs."""

    if patch_yaml:
        monkeypatch.setattr(module.yaml, "safe_load", lambda *_args, **_kwargs: {})

    monkeypatch.setattr(
        "library.logging_utils.configure_logging", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "library.cli_common.ensure_output_dir", lambda path: Path(path)
    )
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
        lambda _path, **_kwargs: (
            tmp_path / "meta.yaml",
            tmp_path / "errors.json",
            tmp_path / "quality",
        ),
    )
    monkeypatch.setattr(
        "library.io.read_ids", lambda *_args, **_kwargs: iter(accessions)
    )
    monkeypatch.setattr("library.io_utils.write_rows", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "library.uniprot_normalize.extract_ensembl_gene_ids",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "library.uniprot_normalize.extract_isoforms",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "library.uniprot_normalize.normalize_entry",
        lambda data, include_seq, isoforms: {
            "uniprot_id": data["primaryAccession"]
        },
    )
    monkeypatch.setattr(
        "library.uniprot_normalize.output_columns",
        lambda *_args, **_kwargs: ["uniprot_id"],
    )
    monkeypatch.setattr(
        "library.http_client.CacheConfig.from_dict",
        lambda *_args, **_kwargs: None,
    )

    iso_calls: list[str] = []
    created_clients: list[Any] = []

    class RecordingClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.fetch_entries_json_calls: list[tuple[tuple[str, ...], int]] = []
            self.fetch_isoforms_fasta_calls: list[str] = []
            created_clients.append(self)

        def fetch_entries_json(
            self, accessions: list[str], *, batch_size: int = 0
        ) -> dict[str, dict[str, str]]:
            call_accessions = tuple(accessions)
            self.fetch_entries_json_calls.append((call_accessions, batch_size))
            return {acc: {"primaryAccession": acc} for acc in call_accessions}

        def fetch_isoforms_fasta(self, accession: str) -> list[str]:
            self.fetch_isoforms_fasta_calls.append(accession)
            iso_calls.append(accession)
            return []

        def fetch_entry_json(self, accession: str) -> dict[str, str]:
            msg = "fetch_entry_json should not be called when batching"
            raise AssertionError(msg)

    monkeypatch.setattr("library.uniprot_client.UniProtClient", RecordingClient)

    return iso_calls, created_clients


def test_uniprot_cli(tmp_path: Path) -> None:
    """CLI should enrich input and write to provided output path."""

    inp = tmp_path / "input.csv"
    inp.write_text(DATA_FILE.read_text())
    out = tmp_path / "out.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(inp),
            "--output",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == str(out)
    assert out.exists()
    df = pd.read_csv(out, dtype=str).fillna("")
    expected_cols = ["uniprot_id", "other"] + list(
        enrich_uniprot.__globals__["OUTPUT_COLUMNS"]
    )
    assert list(df.columns) == expected_cols


def test_get_uniprot_target_data_batches_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`get_uniprot_target_data` should reuse batch responses from UniProt."""

    module = importlib.import_module("scripts.get_uniprot_target_data")

    accessions = [
        f"P{index:05d}" for index in range(module.BATCH_SIZE * 2 + 5)
    ]
    input_path = tmp_path / "input.csv"
    input_path.write_text(
        "uniprot_id\n" + "\n".join(accessions) + "\n", encoding="utf-8"
    )
    output_path = tmp_path / "output.csv"
    iso_output_path = tmp_path / "isoforms.csv"

    iso_calls, created_clients = _monkeypatch_uniprot_cli(
        module, monkeypatch, tmp_path, accessions, patch_yaml=True
    )
    monkeypatch.setattr(
        module,
        "load_uniprot_target_config",
        lambda *_args, **_kwargs: DummyConfig(),
    )

    module.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--with-isoforms",
            "--isoforms-output",
            str(iso_output_path),
        ]
    )

    assert len(created_clients) == 1
    client = created_clients[0]
    expected_batches = math.ceil(len(accessions) / module.BATCH_SIZE)
    assert len(client.fetch_entries_json_calls) == expected_batches
    assert expected_batches < len(accessions)
    for call_accessions, batch_size in client.fetch_entries_json_calls:
        assert batch_size == module.BATCH_SIZE
        assert len(call_accessions) <= module.BATCH_SIZE
    assert iso_calls == accessions


def test_get_uniprot_target_data_custom_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLI should accept explicit config paths and environment overrides."""

    module = importlib.import_module("scripts.get_uniprot_target_data")

    accessions = ["P12345"]
    iso_calls, created_clients = _monkeypatch_uniprot_cli(
        module, monkeypatch, tmp_path, accessions, patch_yaml=False
    )

    recorded_paths: list[Path] = []

    def _recording_loader(path: Path) -> DummyConfig:
        recorded_paths.append(Path(path))
        return DummyConfig()

    monkeypatch.setattr(module, "load_uniprot_target_config", _recording_loader)

    input_path = tmp_path / "input.csv"
    input_path.write_text("uniprot_id\nP12345\n", encoding="utf-8")
    output_path = tmp_path / "output.csv"
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        "output: {}\nuniprot: {}\northologs: {}\nhttp_cache: {}\n",
        encoding="utf-8",
    )

    module.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
        ]
    )

    assert recorded_paths == [config_path.resolve()]
    assert len(created_clients) == 1
    assert iso_calls == []

    recorded_paths.clear()

    env_config = tmp_path / "env.yaml"
    env_config.write_text(
        "output: {}\nuniprot: {}\northologs: {}\nhttp_cache: {}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(module.CONFIG_ENV_VAR, str(env_config))

    module.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(tmp_path / "output_env.csv"),
        ]
    )

    assert recorded_paths == [env_config.resolve()]
