from __future__ import annotations

from pathlib import Path
import json
import logging
import pytest

from chembl2uniprot.config import load_and_validate_config
from library.pipeline_config import (
    load_hgnc_settings,
    load_orthologs_settings,
    load_pipeline_settings,
)

DATA_DIR = Path(__file__).parent / "data"
CONFIG_DIR = DATA_DIR / "config"
SCHEMA = CONFIG_DIR / "config.schema.json"
CONFIG = CONFIG_DIR / "valid.yaml"
PIPELINE_CONFIG = CONFIG_DIR / "pipeline_sections.yaml"


def _write_config(tmp_path: Path, text: str) -> Path:
    cfg = tmp_path / "config.yaml"
    schema = tmp_path / "config.schema.json"
    cfg.write_text(text)
    schema.write_text(SCHEMA.read_text())
    return cfg


def test_invalid_type(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    cfg_text = CONFIG.read_text().replace("rps: 1000", 'rps: "fast"')
    cfg = _write_config(tmp_path, cfg_text)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError):
            load_and_validate_config(cfg)
    assert "uniprot.rate_limit.rps" in caplog.text
    assert "'fast'" in caplog.text
    assert "is not of type 'number'" in caplog.text


def test_unknown_key(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    cfg_text = CONFIG.read_text() + "\nunknown: 42\n"
    cfg = _write_config(tmp_path, cfg_text)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError):
            load_and_validate_config(cfg)
    assert "unknown" in caplog.text
    assert "42" in caplog.text
    assert "Additional properties are not allowed" in caplog.text


def test_invalid_value(tmp_path: Path) -> None:
    cfg_text = CONFIG.read_text().replace("timeout_sec: 30", "timeout_sec: 0")
    cfg = _write_config(tmp_path, cfg_text)
    with pytest.raises(ValueError):
        load_and_validate_config(cfg)


def test_target_chembl_id_alias() -> None:
    """Configuration accepts ``target_chembl_id`` as an alias."""
    cfg = CONFIG_DIR / "alias.yaml"
    loaded = load_and_validate_config(cfg)
    assert loaded.columns.chembl_id == "chembl_id"


def test_chembl_id_with_legacy_schema(tmp_path: Path) -> None:
    """Config using ``chembl_id`` passes against a schema requiring ``target_chembl_id``."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(CONFIG.read_text())
    legacy_schema = json.loads(SCHEMA.read_text())
    columns_schema = legacy_schema["properties"]["columns"]
    columns_schema["properties"].pop("chembl_id")
    columns_schema["required"] = ["target_chembl_id", "uniprot_out"]
    columns_schema.pop("anyOf", None)
    schema_path = tmp_path / "config.schema.json"
    schema_path.write_text(json.dumps(legacy_schema))
    loaded = load_and_validate_config(cfg_path, schema_path)
    assert loaded.columns.chembl_id == "target_chembl_id"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables override YAML configuration values."""
    monkeypatch.setenv("CHEMBL_BATCH__SIZE", "5")
    loaded = load_and_validate_config(CONFIG)
    assert loaded.batch.size == 5


def test_load_pipeline_settings() -> None:
    """Pipeline section is parsed into a strongly typed model."""

    settings = load_pipeline_settings(PIPELINE_CONFIG)
    assert settings.rate_limit_rps == 4.0
    assert settings.include_isoforms is True
    assert settings.iuphar.affinity_parameter == "pIC50"
    assert settings.columns == ["a", "b"]


def test_load_pipeline_settings_defaults(tmp_path: Path) -> None:
    """Missing pipeline section falls back to default values."""

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("{}")
    settings = load_pipeline_settings(cfg_path)
    assert settings.rate_limit_rps == 2.0
    assert settings.columns == []


def test_load_hgnc_settings(tmp_path: Path) -> None:
    """HGNC configuration requires the section to be present."""

    settings = load_hgnc_settings(PIPELINE_CONFIG)
    assert settings.hgnc.base_url.endswith("fetch/uniprot_ids")
    assert settings.output.sep == ";"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("pipeline: {}\n")
    with pytest.raises(KeyError):
        load_hgnc_settings(cfg_path, section="hgnc")


def test_load_orthologs_settings() -> None:
    """Orthologs section falls back to defaults when absent."""

    settings = load_orthologs_settings(PIPELINE_CONFIG)
    assert settings.enabled is False
    assert settings.primary_source == "oma"
    defaults = load_orthologs_settings(CONFIG)
    assert defaults.enabled is True
    assert defaults.target_species == []
