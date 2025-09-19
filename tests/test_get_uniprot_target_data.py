from __future__ import annotations

import importlib
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

import pytest
import requests
import yaml

from library.uniprot_client import UniProtRequestError

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "get_uniprot_target_data.py"


class StubCacheConfig:
    """Minimal cache configuration used to stub project settings."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        path: str | None = None,
        ttl_sec: float = 0.0,
    ) -> None:
        self.enabled = enabled
        self.path = path
        self.ttl_sec = ttl_sec

    def to_cache_dict(self) -> dict[str, Any]:
        """Return dictionary representation compatible with the CLI."""

        return {
            "enabled": self.enabled,
            "path": self.path,
            "ttl_sec": self.ttl_sec,
        }


class StubOutputConfig:
    """Output options emulating the configuration model."""

    def __init__(
        self,
        *,
        sep: str = ",",
        encoding: str = "utf-8",
        list_format: str = "json",
        include_sequence: bool = False,
    ) -> None:
        self.sep = sep
        self.encoding = encoding
        self.list_format = list_format
        self.include_sequence = include_sequence


class StubUniProtConfig:
    """Subset of UniProt configuration required by the CLI."""

    def __init__(
        self,
        *,
        include_isoforms: bool = False,
        use_fasta_stream_for_isoform_ids: bool = False,
        cache: StubCacheConfig | None = None,
        base_url: str = "https://example.uniprot",
        timeout_sec: float = 1.0,
        retries: int = 0,
        rps: float = 10.0,
        batch_size: int = 100,
        fields: Iterable[str] | None = None,
        columns: Iterable[str] | None = None,
    ) -> None:
        self.include_isoforms = include_isoforms
        self.use_fasta_stream_for_isoform_ids = use_fasta_stream_for_isoform_ids
        self.cache = cache
        self.base_url = base_url
        self.timeout_sec = timeout_sec
        self.retries = retries
        self.rps = rps
        self.batch_size = batch_size
        self.fields = list(fields or [])
        self.columns = list(columns or [])

    def model_dump(self) -> dict[str, list[str]]:
        """Return serialisable payload consumed by field resolution helper."""

        return {"fields": self.fields, "columns": self.columns}


class StubOrthologsConfig:
    """Minimal ortholog configuration keeping the CLI happy."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        cache: StubCacheConfig | None = None,
        timeout_sec: float = 1.0,
        retries: int = 0,
        backoff_base_sec: float = 1.0,
        rate_limit_rps: float = 1.0,
        target_species: Iterable[str] | None = None,
    ) -> None:
        self.enabled = enabled
        self.cache = cache
        self.timeout_sec = timeout_sec
        self.retries = retries
        self.backoff_base_sec = backoff_base_sec
        self.rate_limit_rps = rate_limit_rps
        self.target_species = list(target_species or [])


class StubConfig:
    """Top-level configuration container for CLI tests."""

    def __init__(
        self,
        *,
        output: StubOutputConfig,
        uniprot: StubUniProtConfig,
        orthologs: StubOrthologsConfig | None = None,
        http_cache: StubCacheConfig | None = None,
    ) -> None:
        self.output = output
        self.uniprot = uniprot
        self.orthologs = orthologs or StubOrthologsConfig()
        self.http_cache = http_cache


def _setup_cli_stubs(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config: StubConfig,
    *,
    normalize_entry: Callable[[dict[str, Any], bool, list[dict[str, Any]]], dict[str, Any]]
    | None = None,
    output_columns: Callable[[bool], list[str]] | None = None,
) -> dict[str, Any]:
    """Install common monkeypatches for the UniProt CLI tests."""

    records: dict[str, Any] = {
        "metadata": [],
        "serialise": [],
        "cache": [],
        "quality": [],
    }

    monkeypatch.setattr(module.yaml, "safe_load", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        module,
        "load_uniprot_target_config",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(
        "library.logging_utils.configure_logging", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "library.cli_common.ensure_output_dir", lambda path: Path(path)
    )

    def fake_serialise_dataframe(
        df: Any, list_format: str, *, inplace: bool = False
    ) -> Any:
        records["serialise"].append(
            {"list_format": list_format, "inplace": inplace, "rows": int(getattr(df, "shape", (0, 0))[0])}
        )
        return df

    monkeypatch.setattr(
        "library.cli_common.serialise_dataframe", fake_serialise_dataframe
    )

    def fake_write_cli_metadata(
        path: Path,
        *,
        status: str = "ok",
        **kwargs: Any,
    ) -> None:
        payload = dict(kwargs)
        payload["status"] = status
        payload["path"] = Path(path)
        records["metadata"].append(payload)

    monkeypatch.setattr(
        "library.cli_common.write_cli_metadata", fake_write_cli_metadata
    )

    def fake_analyze_table_quality(*_args: Any, **_kwargs: Any) -> None:
        records["quality"].append({})

    monkeypatch.setattr(
        "library.cli_common.analyze_table_quality", fake_analyze_table_quality
    )

    quality_dir = tmp_path / "quality"
    quality_dir.mkdir()
    errors_path = tmp_path / "errors.json"

    def fake_resolve_cli_sidecar_paths(path: Path) -> tuple[Path, Path, Path]:
        return (path.with_suffix(path.suffix + ".meta.yaml"), errors_path, quality_dir)

    monkeypatch.setattr(
        "library.cli_common.resolve_cli_sidecar_paths", fake_resolve_cli_sidecar_paths
    )
    monkeypatch.setattr("library.io_utils.write_rows", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "library.uniprot_normalize.extract_ensembl_gene_ids", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        "library.uniprot_normalize.extract_isoforms", lambda *_args, **_kwargs: []
    )

    def default_normalize_entry(
        data: dict[str, Any], include_seq: bool, isoforms: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "uniprot_id": data["primaryAccession"],
            "gene_primary": "GENE",
            "organism_name": "Human",
            "isoforms_json": "[]",
        }

    monkeypatch.setattr(
        "library.uniprot_normalize.normalize_entry",
        normalize_entry or default_normalize_entry,
    )
    monkeypatch.setattr(
        "library.uniprot_normalize.output_columns",
        output_columns or (lambda *_args, **_kwargs: ["uniprot_id", "gene_primary"]),
    )

    def fake_cache_from_dict(data: Any) -> dict[str, Any] | None:
        if data is None:
            records["cache"].append(None)
            return None
        records["cache"].append(dict(data))
        return {"config": dict(data)}

    monkeypatch.setattr(
        "library.http_client.CacheConfig.from_dict", fake_cache_from_dict
    )

    return records


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


def test_missing_input_file_exits_with_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
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


def test_cli_respects_configuration_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLI should honour configuration defaults when CLI flags are omitted."""

    module = importlib.import_module("scripts.get_uniprot_target_data")

    config = StubConfig(
        output=StubOutputConfig(
            sep=";",
            encoding="utf-16",
            list_format="pipe",
            include_sequence=True,
        ),
        uniprot=StubUniProtConfig(
            include_isoforms=False,
            cache=StubCacheConfig(enabled=True, path="/tmp/uniprot", ttl_sec=120.0),
            base_url="https://api.custom.uniprot",
            timeout_sec=12.5,
            retries=4,
            rps=6.5,
            batch_size=17,
            fields=["accession", "protein_name"],
            columns=["accession"],
        ),
        orthologs=StubOrthologsConfig(enabled=False),
        http_cache=StubCacheConfig(enabled=True, path="/tmp/global", ttl_sec=300.0),
    )

    include_sequence_flags: list[bool] = []

    def recording_normalise_entry(
        data: dict[str, Any], include_seq: bool, isoforms: list[dict[str, Any]]
    ) -> dict[str, Any]:
        include_sequence_flags.append(include_seq)
        return {
            "uniprot_id": data["primaryAccession"],
            "gene_primary": "GENE1",
            "organism_name": "Human",
            "isoforms_json": "[]",
        }

    def configured_output_columns(include_seq: bool) -> list[str]:
        base = ["uniprot_id", "gene_primary"]
        return [*base, "sequence"] if include_seq else base

    records = _setup_cli_stubs(
        module,
        monkeypatch,
        tmp_path,
        config,
        normalize_entry=recording_normalise_entry,
        output_columns=configured_output_columns,
    )

    input_csv = tmp_path / "input.csv"
    input_csv.write_text("uniprot_id\nP12345\n", encoding="utf-8")
    output_csv = tmp_path / "output.csv"

    captured_cfgs: list[Any] = []

    def fake_read_ids(path: Path, column: str, cfg: Any) -> Iterable[str]:
        assert path == input_csv
        assert column == "uniprot_id"
        captured_cfgs.append(cfg)
        yield "P12345"

    monkeypatch.setattr("library.io.read_ids", fake_read_ids)

    field_payloads: list[dict[str, Any]] = []

    def fake_resolve_fields(payload: dict[str, Any]) -> str:
        field_payloads.append(payload)
        return "accession,protein_name"

    monkeypatch.setattr(module, "_resolve_uniprot_fields", fake_resolve_fields)

    client_calls: dict[str, Any] = {"batches": []}

    class RecordingClient:
        def __init__(
            self,
            *,
            base_url: str,
            fields: str,
            network: Any,
            rate_limit: Any,
            cache: Any,
        ) -> None:
            client_calls["base_url"] = base_url
            client_calls["fields"] = fields
            client_calls["network"] = network
            client_calls["rate_limit"] = rate_limit
            client_calls["cache"] = cache

        def fetch_entries_json(
            self, accessions: list[str], *, batch_size: int = 0
        ) -> dict[str, dict[str, Any]]:
            client_calls["batches"].append((tuple(accessions), batch_size))
            return {
                acc: {"primaryAccession": acc, "organism": {"scientificName": "Human"}}
                for acc in accessions
            }

        def fetch_isoforms_fasta(self, *_args: Any, **_kwargs: Any) -> list[str]:
            raise AssertionError("Isoform fetching should be disabled by config")

        def fetch_entry_json(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise AssertionError("Single-entry fetch should not occur in this test")

    monkeypatch.setattr("library.uniprot_client.UniProtClient", RecordingClient)

    module.main(["--input", str(input_csv), "--output", str(output_csv)])

    assert output_csv.exists()
    assert captured_cfgs
    cfg = captured_cfgs[0]
    assert cfg.sep == ";"
    assert cfg.encoding == "utf-16"
    assert cfg.list_format == "pipe"
    assert include_sequence_flags == [True]
    assert field_payloads == [
        {"fields": config.uniprot.fields, "columns": config.uniprot.columns}
    ]

    assert client_calls["base_url"] == "https://api.custom.uniprot"
    network_cfg = client_calls["network"]
    assert network_cfg.timeout_sec == pytest.approx(12.5)
    assert network_cfg.max_retries == 4
    rate_cfg = client_calls["rate_limit"]
    assert rate_cfg.rps == pytest.approx(6.5)
    assert client_calls["cache"] == {"config": config.uniprot.cache.to_cache_dict()}
    assert client_calls["batches"] == [(("P12345",), 17)]
    assert records["cache"][0] == config.http_cache.to_cache_dict()
    assert records["cache"][1] == config.uniprot.cache.to_cache_dict()
    assert records["serialise"][0]["list_format"] == "pipe"
    assert records["serialise"][0]["inplace"] is True
    assert records["metadata"][0]["status"] == "ok"


def test_cli_uses_custom_batch_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLI should respect ``--batch-size`` overrides when batching accessions."""

    module = importlib.import_module("scripts.get_uniprot_target_data")

    config = StubConfig(
        output=StubOutputConfig(),
        uniprot=StubUniProtConfig(batch_size=module.DEFAULT_BATCH_SIZE),
    )

    records = _setup_cli_stubs(module, monkeypatch, tmp_path, config)

    custom_batch_size = 3
    accessions = [f"P{index:05d}" for index in range(custom_batch_size * 2 + 1)]

    input_csv = tmp_path / "input.csv"
    input_csv.write_text(
        "uniprot_id\n" + "\n".join(accessions) + "\n",
        encoding="utf-8",
    )
    output_csv = tmp_path / "output.csv"

    def fake_read_ids(path: Path, column: str, cfg: Any) -> Iterable[str]:
        assert path == input_csv
        assert column == "uniprot_id"
        yield from accessions

    monkeypatch.setattr("library.io.read_ids", fake_read_ids)
    monkeypatch.setattr(module, "_resolve_uniprot_fields", lambda *_args, **_kwargs: "id")

    fetch_calls: list[tuple[tuple[str, ...], int]] = []

    class RecordingClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def fetch_entries_json(
            self, accessions: list[str], *, batch_size: int = 0
        ) -> dict[str, dict[str, Any]]:
            fetch_calls.append((tuple(accessions), batch_size))
            return {acc: {"primaryAccession": acc} for acc in accessions}

        def fetch_isoforms_fasta(self, *_args: Any, **_kwargs: Any) -> list[str]:
            raise AssertionError("Isoform fetching should remain disabled")

        def fetch_entry_json(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise AssertionError("Single entry fetches are unexpected")

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
    assert records["metadata"][0]["status"] == "ok"


def test_cli_reports_network_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLI should convert UniProt client failures into a clean exit."""

    module = importlib.import_module("scripts.get_uniprot_target_data")

    config = StubConfig(
        output=StubOutputConfig(),
        uniprot=StubUniProtConfig(),
    )

    records = _setup_cli_stubs(module, monkeypatch, tmp_path, config)

    input_csv = tmp_path / "input.csv"
    input_csv.write_text("uniprot_id\nP12345\n", encoding="utf-8")
    output_csv = tmp_path / "output.csv"

    monkeypatch.setattr(module, "_resolve_uniprot_fields", lambda *_args, **_kwargs: "id")

    def fake_read_ids(path: Path, column: str, cfg: Any) -> Iterable[str]:
        yield "P12345"

    monkeypatch.setattr("library.io.read_ids", fake_read_ids)

    class FailingClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def fetch_entries_json(
            self, *_args: Any, **_kwargs: Any
        ) -> dict[str, dict[str, Any]]:
            raise UniProtRequestError(
                "https://api.custom.uniprot",
                attempts=2,
                cause=requests.RequestException("boom"),
            )

        def fetch_isoforms_fasta(self, *_args: Any, **_kwargs: Any) -> list[str]:
            return []

        def fetch_entry_json(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {}

    monkeypatch.setattr("library.uniprot_client.UniProtClient", FailingClient)

    with pytest.raises(SystemExit) as excinfo:
        module.main(["--input", str(input_csv), "--output", str(output_csv)])

    assert excinfo.value.code == 2
    assert records["metadata"]
    failure_meta = records["metadata"][0]
    assert failure_meta["status"] == "error"
    assert "Failed to retrieve" in failure_meta["error"]
    assert records["serialise"] == []
    assert records["quality"] == []
