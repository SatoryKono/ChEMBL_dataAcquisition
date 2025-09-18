"""Tests for the :mod:`chembl2uniprot` command line interface."""

from __future__ import annotations

import csv
import itertools
import math
import tracemalloc
from pathlib import Path

import pandas as pd
import pytest

import chembl2uniprot_main

from chembl2uniprot.mapping import BatchMappingResult

SCHEMA_SOURCE = Path(__file__).parent / "data" / "config" / "config.schema.json"
CONFIG_TEMPLATE = """
io:
  input:
    encoding: "utf-8"
  output:
    encoding: "utf-8"
  csv:
    separator: ","
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

    config_path.write_text(
        CONFIG_TEMPLATE.format(batch_size=batch_size), encoding="utf-8"
    )
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
    output_path = tmp_path / "output.csv"
    config_path = tmp_path / "config.yaml"
    schema_path = tmp_path / "config.schema.json"

    config_path.write_text(CONFIG_TEMPLATE.format(batch_size=10), encoding="utf-8")
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

    assert rows == [
        {"target_chembl_id": "CHEMBL1", "mapped_uniprot_id": "UP_CHEMBL1"},
        {"target_chembl_id": "CHEMBL2", "mapped_uniprot_id": "UP_CHEMBL2"},
    ]


def test_cli_empty_input_skips_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    config_path = tmp_path / "config.yaml"
    schema_path = tmp_path / "config.schema.json"

    config_path.write_text(CONFIG_TEMPLATE.format(batch_size=5), encoding="utf-8")
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
