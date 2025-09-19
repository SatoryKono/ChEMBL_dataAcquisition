from __future__ import annotations

import subprocess
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any

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


def test_cli_records_warning_when_ortholog_flags_conflict(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ensure the metadata captures ortholog preference conflicts."""

    loader = SourceFileLoader("get_uniprot_target_data_conflict", str(SCRIPT))
    module = loader.load_module()

    input_path = tmp_path / "input.csv"
    input_path.write_text("uniprot_id\nP12345\n", encoding="utf-8")
    output_path = tmp_path / "output.csv"

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
            self.rps = 5.0
            self.fields: list[str] = []
            self.columns: list[str] = []

        def model_dump(self) -> dict[str, list[str]]:
            return {"fields": self.fields, "columns": self.columns}

    class DummyOrthologsConfig:
        def __init__(self) -> None:
            self.enabled = True
            self.cache = None
            self.timeout_sec = 1.0
            self.retries = 0
            self.backoff_base_sec = 1.0
            self.rate_limit_rps = 2.0

    class DummyConfig:
        def __init__(self) -> None:
            self.http_cache = None
            self.output = DummyOutputConfig()
            self.uniprot = DummyUniProtConfig()
            self.orthologs = DummyOrthologsConfig()

    monkeypatch.setattr(
        module,
        "load_uniprot_target_config",
        lambda *_args, **_kwargs: DummyConfig(),
    )
    monkeypatch.setattr(
        "library.cli_common.analyze_table_quality", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "library.cli_common.serialise_dataframe",
        lambda df, list_format, *, inplace=False: df,
    )
    monkeypatch.setattr(
        "library.http_client.CacheConfig.from_dict", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "library.uniprot_normalize.extract_ensembl_gene_ids",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "library.uniprot_normalize.normalize_entry",
        lambda data, include_seq, isoforms: {"uniprot_id": data["primaryAccession"]},
    )
    monkeypatch.setattr(
        "library.uniprot_normalize.output_columns",
        lambda *_args, **_kwargs: ["uniprot_id"],
    )

    class DummyClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def fetch_entries_json(
            self, accessions: list[str], *, batch_size: int = 0
        ) -> dict[str, dict[str, str]]:
            return {
                accession: {"primaryAccession": accession} for accession in accessions
            }

        def fetch_isoforms_fasta(self, _accession: str) -> list[str]:
            return []

    monkeypatch.setattr("library.uniprot_client.UniProtClient", DummyClient)

    expected_warning = (
        "Ortholog enrichment configuration conflict: CLI requested False but "
        "configuration orthologs.enabled=True. Ortholog data will not be "
        "retrieved. Use --orthologs-enabled/--no-orthologs-enabled to override "
        "the configuration."
    )

    module.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--no-with-orthologs",
        ]
    )

    captured = capsys.readouterr()
    assert expected_warning in captured.err

    meta_path = output_path.with_name(f"{output_path.name}.meta.yaml")
    assert meta_path.exists()
    metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    assert metadata["warnings"] == [expected_warning]
