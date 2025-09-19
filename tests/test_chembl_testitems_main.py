from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

chembl_testitems_main = importlib.import_module("scripts.chembl_testitems_main")


def test_parse_args_rejects_dictionary() -> None:
    with pytest.raises(SystemExit):
        chembl_testitems_main.parse_args(["--dictionary", "dict.csv"])


def test_run_pipeline_passes_pubchem_http_client_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = chembl_testitems_main

    input_csv = tmp_path / "input.csv"
    input_csv.write_text("molecule_chembl_id\nCHEMBL1\n", encoding="utf-8")

    output_csv = tmp_path / "out.csv"

    captured: dict[str, Any] = {}

    class DummyClient:  # pragma: no cover - simple stub
        def __init__(self, **kwargs: Any) -> None:
            captured["chembl_client"] = kwargs

    def fake_get_testitems(
        _client: Any, molecule_ids: Any, *, chunk_size: int
    ) -> pd.DataFrame:
        captured["chunk_size"] = chunk_size
        ids = list(molecule_ids)
        return pd.DataFrame(
            {
                "molecule_chembl_id": ids,
                "canonical_smiles": ["C"] * len(ids),
            }
        )

    def fake_normalize(df: pd.DataFrame) -> pd.DataFrame:
        return df

    def fake_add_pubchem_data(
        df: pd.DataFrame,
        *,
        smiles_column: str,
        timeout: float,
        base_url: str,
        user_agent: str,
        http_client_config: dict[str, Any] | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        captured["pubchem_kwargs"] = {
            "smiles_column": smiles_column,
            "timeout": timeout,
            "base_url": base_url,
            "user_agent": user_agent,
        }
        captured["http_client_config"] = http_client_config
        return df

    def fake_validate(
        df: pd.DataFrame,
        schema: Any = None,
        *,
        errors_path: Path,
    ) -> pd.DataFrame:
        _ = schema
        captured["errors_path"] = errors_path
        return df

    def fake_write_cli_metadata(
        output_path: Path,
        *,
        row_count: int,
        column_count: int,
        namespace: Any,
        command_parts: Sequence[str] | None = None,
        meta_path: Path | None = None,
    ) -> None:
        captured["meta_output_path"] = output_path
        captured["meta_command"] = " ".join(command_parts or [])
        captured["meta_config"] = vars(namespace)

        captured["meta_row_count"] = row_count
        captured["meta_column_count"] = column_count
        captured["meta_path"] = meta_path
        return output_path.with_name(f"{output_path.name}.meta.yaml")

    monkeypatch.setattr(module, "ChemblClient", DummyClient)
    monkeypatch.setattr(module, "get_testitems", fake_get_testitems)
    monkeypatch.setattr(module, "normalize_testitems", fake_normalize)
    monkeypatch.setattr(module, "add_pubchem_data", fake_add_pubchem_data)
    monkeypatch.setattr(module, "validate_testitems", fake_validate)

    monkeypatch.setattr(module, "write_cli_metadata", fake_write_cli_metadata)

    monkeypatch.setattr(module, "analyze_table_quality", lambda *_, **__: None)

    argv = [
        "--input",
        str(input_csv),
        "--output",
        str(output_csv),
        "--pubchem-max-retries",
        "9",
        "--pubchem-rps",
        "1.5",
        "--pubchem-backoff",
        "2.0",
        "--pubchem-retry-penalty",
        "7.5",
    ]
    args = module.parse_args(argv)

    exit_code = module.run_pipeline(
        args,
        command_parts=["chembl_testitems_main.py", *argv],
    )

    assert exit_code == 0
    assert output_csv.exists()
    assert captured["http_client_config"] == {
        "max_retries": 9,
        "rps": 1.5,
        "backoff_multiplier": 2.0,
        "retry_penalty_seconds": 7.5,
    }
    assert captured["meta_config"]["pubchem_max_retries"] == 9
    assert captured["meta_config"]["pubchem_rps"] == 1.5
    assert captured["meta_config"]["pubchem_backoff"] == 2.0
    assert captured["meta_config"]["pubchem_retry_penalty"] == 7.5
    assert captured["chunk_size"] == args.chunk_size


def test_run_pipeline_adds_required_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = chembl_testitems_main

    input_csv = tmp_path / "input.csv"
    input_csv.write_text("molecule_chembl_id\nCHEMBL1\n", encoding="utf-8")

    output_csv = tmp_path / "out.csv"

    class DummyClient:  # pragma: no cover - simple stub
        def __init__(self, **_: Any) -> None:  # noqa: D401 - signature compatibility
            """Ignore initialisation arguments for the dummy client."""

    def fake_get_testitems(
        _client: Any, molecule_ids: Any, *, chunk_size: int
    ) -> pd.DataFrame:
        _ = chunk_size
        ids = list(molecule_ids)
        return pd.DataFrame({"molecule_chembl_id": ids})

    def fake_normalize(df: pd.DataFrame) -> pd.DataFrame:
        return df

    def fake_add_pubchem_data(
        df: pd.DataFrame,
        *,
        smiles_column: str,
        timeout: float,
        base_url: str,
        user_agent: str,
        http_client_config: dict[str, Any] | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        _ = smiles_column, timeout, base_url, user_agent, http_client_config
        return df

    def fake_validate(
        df: pd.DataFrame,
        schema: Any = None,
        *,
        errors_path: Path,
    ) -> pd.DataFrame:
        _ = schema, errors_path
        return df

    monkeypatch.setattr(module, "ChemblClient", DummyClient)
    monkeypatch.setattr(module, "get_testitems", fake_get_testitems)
    monkeypatch.setattr(module, "normalize_testitems", fake_normalize)
    monkeypatch.setattr(module, "add_pubchem_data", fake_add_pubchem_data)
    monkeypatch.setattr(module, "validate_testitems", fake_validate)
    monkeypatch.setattr(module, "analyze_table_quality", lambda *_, **__: None)
    monkeypatch.setattr(module, "write_cli_metadata", lambda *_, **__: None)

    exit_code = module.run_pipeline(
        module.parse_args(
            [
                "--input",
                str(input_csv),
                "--output",
                str(output_csv),
            ]
        ),
        command_parts=["chembl_testitems_main.py", "--input", str(input_csv)],
    )

    assert exit_code == 0
    assert output_csv.exists()
    df = pd.read_csv(output_csv)
    for column in module.REQUIRED_ENRICHED_COLUMNS:
        assert column in df.columns
