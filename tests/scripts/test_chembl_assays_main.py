from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pytest
from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

chembl_assays_main = importlib.import_module("scripts.chembl_assays_main")
cli_common = importlib.import_module("library.cli_common")


def test_run_pipeline_preserves_numeric_dtypes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = chembl_assays_main

    input_csv = tmp_path / "input.csv"
    input_csv.write_text("assay_chembl_id\nCHEMBL1\n", encoding="utf-8")
    output_csv = tmp_path / "out.csv"

    validated_frame = pd.DataFrame(
        {
            "assay_chembl_id": ["CHEMBL1"],
            "document_chembl_id": ["DOC1"],
            "target_chembl_id": ["TAR1"],
            "assay_with_same_target": pd.Series([2], dtype="int64"),
            "confidence_score": pd.Series([7], dtype="int64"),
            "pchembl_value": pd.Series([7.5], dtype="float64"),
            "is_active": pd.Series([True], dtype="bool"),
            "assay_parameters": [[{"name": "Param", "value": 1}]],
        }
    )

    captured: dict[str, Any] = {}
    dtypes_capture: dict[str, pd.Series] = {}

    class DummyClient:  # pragma: no cover - simple stub
        def __init__(self, **_: Any) -> None:
            """Ignore client configuration arguments."""

    def fake_read_ids(*_: object) -> list[str]:
        return ["CHEMBL1"]

    def fake_get_assays(_client: Any, *_: object, **__: object) -> pd.DataFrame:
        return validated_frame.copy()

    def fake_postprocess(df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def fake_normalize(df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def fake_validate(
        df: pd.DataFrame,
        schema: Any | None = None,
        *,
        errors_path: Path,
    ) -> pd.DataFrame:
        _ = schema
        captured["errors_path"] = errors_path
        return df.copy()

    original_serialise = cli_common.serialise_dataframe

    def serialise_spy(
        df: pd.DataFrame, list_format: str, *, inplace: bool = False
    ) -> pd.DataFrame:
        captured["serialise_call"] = {
            "inplace": inplace,
            "list_format": list_format,
            "input_id": id(df),
        }
        result = original_serialise(df, list_format, inplace=inplace)
        captured["serialise_call"]["result_id"] = id(result)
        return result

    def fake_analyze_table_quality(frame: pd.DataFrame, table_name: str) -> None:
        captured["quality_table"] = table_name
        dtypes_capture["values"] = frame.dtypes.copy()

    def fake_write_cli_metadata(
        output_path: Path,
        *,
        row_count: int,
        column_count: int,
        namespace: Any,
        command_parts: Sequence[str] | None = None,
        meta_path: Path | None = None,
        status: str = "success",
        error: str | None = None,
        warnings: Sequence[str] | None = None,
    ) -> Path:
        captured["metadata"] = {
            "command_parts": tuple(command_parts or ()),
            "namespace": namespace,
            "row_count": row_count,
            "column_count": column_count,
            "meta_path": meta_path,
            "status": status,
            "error": error,
            "warnings": list(warnings or []),
        }
        return output_path.with_name(f"{output_path.name}.meta.yaml")

    monkeypatch.setattr(module, "ChemblClient", DummyClient)
    monkeypatch.setattr(module, "read_ids", fake_read_ids)
    monkeypatch.setattr(module, "get_assays", fake_get_assays)
    monkeypatch.setattr(module, "postprocess_assays", fake_postprocess)
    monkeypatch.setattr(module, "normalize_assays", fake_normalize)
    monkeypatch.setattr(module, "validate_assays", fake_validate)
    monkeypatch.setattr(module, "write_cli_metadata", fake_write_cli_metadata)
    monkeypatch.setattr(module, "analyze_table_quality", fake_analyze_table_quality)
    monkeypatch.setattr(cli_common, "serialise_dataframe", serialise_spy)
    monkeypatch.setattr(module, "serialise_dataframe", serialise_spy)

    argv = [
        "--input",
        str(input_csv),
        "--output",
        str(output_csv),
        "--base-url",
        "https://example.org/api",
    ]
    args = module.parse_args(argv)

    exit_code = module.run_pipeline(
        args,
        command_parts=["chembl_assays_main.py", *argv],
    )

    assert exit_code == 0
    assert output_csv.exists()
    assert captured["serialise_call"]["inplace"] is True
    assert captured["serialise_call"]["list_format"] == args.list_format
    assert (
        captured["serialise_call"]["input_id"]
        == captured["serialise_call"]["result_id"]
    )

    dtypes = dtypes_capture["values"]
    assert is_integer_dtype(dtypes["assay_with_same_target"])
    assert is_integer_dtype(dtypes["confidence_score"])
    assert is_float_dtype(dtypes["pchembl_value"])
    assert is_bool_dtype(dtypes["is_active"])

    metadata = captured["metadata"]
    assert metadata["namespace"] is args
    assert metadata["command_parts"] == tuple(["chembl_assays_main.py", *argv])
    assert metadata["row_count"] == 1
    assert metadata["column_count"] == len(validated_frame.columns)
