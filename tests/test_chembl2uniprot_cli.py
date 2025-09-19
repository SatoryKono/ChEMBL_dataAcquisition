"""Tests for the :mod:`chembl2uniprot` command line interface."""

from __future__ import annotations

import csv
import itertools
import math
import tracemalloc
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

import chembl2uniprot_main

from chembl2uniprot.mapping import BatchMappingResult

SCHEMA_SOURCE = Path(__file__).parent / "data" / "config" / "config.schema.json"
CONFIG_TEMPLATE = """
io:
  input:
    encoding: "{input_encoding}"
  output:
    encoding: "{output_encoding}"
  csv:
    separator: "{separator}"
    multivalue_delimiter: "|"
columns:
  chembl_id: "target_chembl_id"
  uniprot_out: "mapped_uniprot_id"
uniprot:
  base_url: "https://rest.uniprot.org"
  id_mapping:
    endpoint: "/idmapping/run"
    status_endpoint: "/idmapping/status"
    results_endpoint: "/idmapping/results"
    from_db: "ChEMBL"
    to_db: "UniProtKB"
  polling:
    interval_sec: 0
  rate_limit:
    rps: 1000
  retry:
    max_attempts: 1
    backoff_sec: 0
network:
  timeout_sec: 30
batch:
  size: {batch_size}
logging:
  level: "ERROR"
  format: "human"
"""


def render_config(
    *,
    batch_size: int,
    separator: str = ",",
    input_encoding: str = "utf-8",
    output_encoding: str | None = None,
) -> str:
    """Return a formatted configuration snippet for the CLI tests."""

    effective_output = (
        output_encoding if output_encoding is not None else input_encoding
    )
    return CONFIG_TEMPLATE.format(
        batch_size=batch_size,
        separator=separator,
        input_encoding=input_encoding,
        output_encoding=effective_output,
    )


def test_cli_handles_large_input_without_memory_spikes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ensure the CLI processes large files lazily and stays within memory limits."""

    total_rows = 100_000
    batch_size = 1_000

    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    config_path = tmp_path / "config.yaml"
    schema_path = tmp_path / "config.schema.json"

    config_path.write_text(render_config(batch_size=batch_size), encoding="utf-8")
    schema_path.write_text(SCHEMA_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")

    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_chembl_id"])
        for idx in range(total_rows):
            writer.writerow([f"CHEMBL{idx}"])

    batch_sizes: list[int] = []
    counter = itertools.count()

    def fake_map_batch(ids, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        batch_sizes.append(len(ids))
        return BatchMappingResult(
            mapping={identifier: [f"UP{next(counter)}"] for identifier in ids},
            failed_ids=[],
        )

    import chembl2uniprot.mapping as mapping_module

    monkeypatch.setattr(mapping_module, "_map_batch", fake_map_batch)
    monkeypatch.setattr(
        mapping_module,
        "analyze_table_quality",
        lambda *args, **kwargs: (None, None),
    )

    def _fail_read_csv(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError(
            "pandas.read_csv should not be invoked during streaming execution"
        )

    monkeypatch.setattr(pd, "read_csv", _fail_read_csv)

    tracemalloc.start()
    chembl2uniprot_main.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
            "--log-level",
            "ERROR",
            "--log-format",
            "human",
            "--sep",
            ",",
            "--encoding",
            "utf-8",
        ]
    )
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    captured = capsys.readouterr()
    assert str(output_path) in captured.out

    assert batch_sizes
    assert max(batch_sizes) <= batch_size
    assert len(batch_sizes) == math.ceil(total_rows / batch_size)

    assert peak < 80 * 1024 * 1024, f"Peak memory too high: {peak} bytes"
    assert current < peak

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert reader.fieldnames[-1] == "mapped_uniprot_id"
        first_rows = [next(reader) for _ in range(3)]

    assert [row["mapped_uniprot_id"] for row in first_rows] == ["UP0", "UP1", "UP2"]


def test_cli_success_runs_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "nested" / "outputs" / "output.csv"
    config_path = tmp_path / "config.yaml"
    schema_path = tmp_path / "config.schema.json"

    config_path.write_text(render_config(batch_size=10), encoding="utf-8")
    schema_path.write_text(SCHEMA_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")

    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_chembl_id"])
        writer.writerow(["CHEMBL1"])
        writer.writerow(["CHEMBL2"])

    def fake_map_batch(ids, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return BatchMappingResult(
            mapping={identifier: [f"UP_{identifier}"] for identifier in ids},
            failed_ids=[],
        )

    import chembl2uniprot.mapping as mapping_module

    quality_calls: list[tuple[object, object, str, str | None]] = []

    def fake_analyze(
        table: object,
        table_name: object,
        separator: str = ",",
        encoding: str | None = None,
    ) -> tuple[None, None]:
        quality_calls.append((table, table_name, separator, encoding))
        return (None, None)

    monkeypatch.setattr(mapping_module, "_map_batch", fake_map_batch)
    monkeypatch.setattr(mapping_module, "analyze_table_quality", fake_analyze)

    assert not output_path.parent.exists()

    chembl2uniprot_main.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
            "--log-level",
            "ERROR",
            "--log-format",
            "human",
            "--sep",
            ",",
            "--encoding",
            "utf-8",
        ]
    )

    captured = capsys.readouterr()
    assert str(output_path) in captured.out

    assert output_path.exists()
    assert output_path.parent.exists()

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert rows == [
        {"target_chembl_id": "CHEMBL1", "mapped_uniprot_id": "UP_CHEMBL1"},
        {"target_chembl_id": "CHEMBL2", "mapped_uniprot_id": "UP_CHEMBL2"},
    ]

    meta_path = output_path.with_name(f"{output_path.name}.meta.yaml")
    assert meta_path.exists()
    metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    assert metadata["rows"] == 2
    assert metadata["columns"] == 2
    assert metadata["config"]["separator"] == ","
    assert metadata["config"]["input_encoding"] == "utf-8"
    assert metadata["config"]["output_encoding"] == "utf-8"
    assert metadata["config"]["log_level"] == "ERROR"
    assert metadata["config"]["log_format"] == "human"
    assert metadata["command"].startswith("map_chembl_to_uniprot(")

    errors_path = output_path.with_name(f"{output_path.name}.errors.json")
    assert not errors_path.exists()

    assert quality_calls, "analyze_table_quality should have been invoked"
    quality_table, table_name, used_sep, used_encoding = quality_calls[-1]
    assert quality_table == output_path
    assert Path(str(table_name)).name == output_path.stem
    assert used_sep == ","
    assert used_encoding == "utf-8"


def test_cli_uses_yaml_separator_and_encoding_when_not_overridden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify the CLI respects YAML separator and encoding without overrides."""

    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    config_path = tmp_path / "config.yaml"
    schema_path = tmp_path / "config.schema.json"

    custom_sep = ";"
    custom_encoding = "iso-8859-1"

    config_path.write_text(
        render_config(
            batch_size=10,
            separator=custom_sep,
            input_encoding=custom_encoding,
            output_encoding=custom_encoding,
        ),
        encoding="utf-8",
    )
    schema_path.write_text(SCHEMA_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")

    with input_path.open("w", encoding=custom_encoding, newline="") as handle:
        writer = csv.writer(handle, delimiter=custom_sep)
        writer.writerow(["target_chembl_id"])
        writer.writerow(["CHEMBLÉ"])

    def fake_map_batch(ids, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return BatchMappingResult(
            mapping={identifier: [f"UP_{identifier}"] for identifier in ids},
            failed_ids=[],
        )

    import chembl2uniprot.mapping as mapping_module

    quality_calls: list[tuple[object, object, str, str | None]] = []

    def fake_analyze(
        table: object,
        table_name: object,
        separator: str = ",",
        encoding: str | None = None,
    ) -> tuple[None, None]:
        quality_calls.append((table, table_name, separator, encoding))
        return (None, None)

    monkeypatch.setattr(mapping_module, "_map_batch", fake_map_batch)
    monkeypatch.setattr(mapping_module, "analyze_table_quality", fake_analyze)

    chembl2uniprot_main.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
        ]
    )

    raw_bytes = output_path.read_bytes()
    assert b";" in raw_bytes
    assert "CHEMBLÉ".encode(custom_encoding) in raw_bytes
    assert "UP_CHEMBLÉ".encode(custom_encoding) in raw_bytes

    with pytest.raises(UnicodeDecodeError):
        output_path.read_text(encoding="utf-8")

    decoded = output_path.read_text(encoding=custom_encoding)
    assert "CHEMBLÉ" in decoded
    assert "UP_CHEMBLÉ" in decoded

    assert quality_calls, "Expected quality analysis to be invoked"
    _, table_name, used_sep, used_encoding = quality_calls[-1]
    assert used_sep == custom_sep
    assert used_encoding == custom_encoding
    assert Path(str(table_name)).name == output_path.stem


def test_cli_writes_failed_identifier_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    config_path = tmp_path / "config.yaml"
    schema_path = tmp_path / "config.schema.json"

    config_path.write_text(render_config(batch_size=5), encoding="utf-8")
    schema_path.write_text(SCHEMA_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")

    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_chembl_id"])
        writer.writerow(["CHEMBL1"])
        writer.writerow(["CHEMBL2"])

    def fake_map_batch(ids, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        mapping: dict[str, list[str]] = {}
        failed: list[str] = []
        for identifier in ids:
            if identifier.endswith("1"):
                mapping[identifier] = [f"UP_{identifier}"]
            else:
                failed.append(identifier)
        return BatchMappingResult(mapping=mapping, failed_ids=failed)

    import chembl2uniprot.mapping as mapping_module

    monkeypatch.setattr(mapping_module, "_map_batch", fake_map_batch)
    monkeypatch.setattr(
        mapping_module, "analyze_table_quality", lambda *args, **kwargs: (None, None)
    )

    chembl2uniprot_main.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
        ]
    )

    errors_path = output_path.with_name(f"{output_path.name}.errors.json")
    assert errors_path.exists()
    payload = json.loads(errors_path.read_text(encoding="utf-8"))
    assert payload["failed_identifiers"] == ["CHEMBL2"]
    assert payload["count"] == 1


def test_cli_clears_stale_error_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    config_path = tmp_path / "config.yaml"
    schema_path = tmp_path / "config.schema.json"

    config_path.write_text(render_config(batch_size=5), encoding="utf-8")
    schema_path.write_text(SCHEMA_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")

    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_chembl_id"])
        writer.writerow(["CHEMBL1"])

    def fake_map_batch(ids, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return BatchMappingResult(
            mapping={identifier: [f"UP_{identifier}"] for identifier in ids},
            failed_ids=[],
        )

    import chembl2uniprot.mapping as mapping_module

    monkeypatch.setattr(mapping_module, "_map_batch", fake_map_batch)
    monkeypatch.setattr(
        mapping_module, "analyze_table_quality", lambda *args, **kwargs: (None, None)
    )

    errors_path = output_path.with_name(f"{output_path.name}.errors.json")
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.write_text(
        json.dumps({"failed_identifiers": ["STALE"], "count": 1}),
        encoding="utf-8",
    )

    chembl2uniprot_main.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
        ]
    )

    assert not errors_path.exists()


def test_cli_empty_input_skips_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    config_path = tmp_path / "config.yaml"
    schema_path = tmp_path / "config.schema.json"

    config_path.write_text(render_config(batch_size=5), encoding="utf-8")
    schema_path.write_text(SCHEMA_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")

    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_chembl_id"])

    def fail_map_batch(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("_map_batch should not be called for empty inputs")

    import chembl2uniprot.mapping as mapping_module

    monkeypatch.setattr(mapping_module, "_map_batch", fail_map_batch)
    monkeypatch.setattr(
        mapping_module, "analyze_table_quality", lambda *args, **kwargs: (None, None)
    )

    chembl2uniprot_main.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
            "--log-level",
            "ERROR",
            "--log-format",
            "human",
            "--sep",
            ",",
            "--encoding",
            "utf-8",
        ]
    )

    captured = capsys.readouterr()
    assert str(output_path) in captured.out

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert rows == []
