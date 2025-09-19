from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chembl2uniprot.config import (  # noqa: E402
    load_and_validate_config,
    resolve_runtime_options,
)
from chembl2uniprot.mapping import (  # noqa: E402
    BatchMappingResult,
    get_ids_from_dataframe,
    map_chembl_to_uniprot,
)

SCHEMA_SOURCE = Path(__file__).parent / "data" / "config" / "config.schema.json"

CONFIG_TEMPLATE = """
contact:
  name: "Test Maintainer"
  email: "maintainer@example.org"
  user_agent: "test-suite/1.0 (mailto:maintainer@example.org)"
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
  level: "INFO"
  format: "human"
"""


def render_config(
    *,
    batch_size: int,
    separator: str = ",",
    input_encoding: str = "utf-8",
    output_encoding: str | None = None,
) -> str:
    """Return a formatted configuration snippet for library tests."""

    effective_output = (
        output_encoding if output_encoding is not None else input_encoding
    )
    return CONFIG_TEMPLATE.format(
        batch_size=batch_size,
        separator=separator,
        input_encoding=input_encoding,
        output_encoding=effective_output,
    )


def test_get_ids_from_dataframe_filters_nan_and_empty_strings() -> None:
    df = pd.DataFrame(
        {
            "chembl_id": [
                "CHEMBL1",
                pd.NA,
                float("nan"),
                "nan",
                " NaN ",
                "",
                "CHEMBL2",
                "CHEMBL1",
            ]
        }
    )

    assert get_ids_from_dataframe(df, "chembl_id") == ["CHEMBL1", "CHEMBL2"]


def test_map_chembl_to_uniprot_uses_yaml_separator_and_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure mapping honours YAML separator and encoding when not overridden."""

    custom_sep = ";"
    custom_encoding = "iso-8859-1"

    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    config_path = tmp_path / "config.yaml"
    schema_path = tmp_path / "config.schema.json"

    config_path.write_text(
        render_config(
            batch_size=5,
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

    monkeypatch.setattr(mapping_module, "_map_batch", fake_map_batch)
    monkeypatch.setattr(
        mapping_module, "analyze_table_quality", lambda *args, **kwargs: (None, None)
    )

    result_path = map_chembl_to_uniprot(
        input_csv_path=input_path,
        output_csv_path=output_path,
        config_path=config_path,
        schema_path=schema_path,
    )

    assert result_path == output_path

    raw_bytes = output_path.read_bytes()
    assert b";" in raw_bytes
    assert "CHEMBLÉ".encode(custom_encoding) in raw_bytes
    assert "UP_CHEMBLÉ".encode(custom_encoding) in raw_bytes

    with pytest.raises(UnicodeDecodeError):
        output_path.read_text(encoding="utf-8")

    decoded = output_path.read_text(encoding=custom_encoding)
    assert "CHEMBLÉ" in decoded
    assert "UP_CHEMBLÉ" in decoded


def test_resolve_runtime_options_combines_cli_and_yaml(tmp_path: Path) -> None:
    """Validate CLI overrides merge correctly with YAML configuration values."""

    config_path = tmp_path / "config.yaml"
    schema_path = tmp_path / "config.schema.json"

    config_path.write_text(
        render_config(
            batch_size=3,
            separator=";",
            input_encoding="windows-1250",
            output_encoding="utf-8",
        ),
        encoding="utf-8",
    )
    schema_path.write_text(SCHEMA_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")

    cfg = load_and_validate_config(config_path, schema_path)

    defaults = resolve_runtime_options(cfg)
    assert defaults.log_level == "INFO"
    assert defaults.log_format == "human"
    assert defaults.separator == ";"
    assert defaults.input_encoding == "windows-1250"
    assert defaults.output_encoding == "utf-8"

    overrides = resolve_runtime_options(
        cfg,
        cli_log_level="DEBUG",
        cli_log_format="json",
        cli_sep="\t",
        cli_encoding="utf-16",
    )
    assert overrides.log_level == "DEBUG"
    assert overrides.log_format == "json"
    assert overrides.separator == "\t"
    assert overrides.input_encoding == "utf-16"
    assert overrides.output_encoding == "utf-16"

    mixed = resolve_runtime_options(cfg, cli_sep="|")
    assert mixed.log_level == "INFO"
    assert mixed.log_format == "human"
    assert mixed.separator == "|"
    assert mixed.input_encoding == "windows-1250"
    assert mixed.output_encoding == "utf-8"
