from __future__ import annotations

from pathlib import Path
import json
import pytest

from chembl2uniprot.config import load_and_validate_config

DATA_DIR = Path(__file__).parent / "data"
CONFIG_DIR = DATA_DIR / "config"
SCHEMA = CONFIG_DIR / "config.schema.json"
CONFIG = CONFIG_DIR / "valid.yaml"


def _write_config(tmp_path: Path, text: str) -> Path:
    cfg = tmp_path / "config.yaml"
    schema = tmp_path / "config.schema.json"
    cfg.write_text(text)
    schema.write_text(SCHEMA.read_text())
    return cfg


def test_invalid_type(tmp_path: Path) -> None:
    cfg_text = CONFIG.read_text().replace("rps: 1000", 'rps: "fast"')
    cfg = _write_config(tmp_path, cfg_text)
    with pytest.raises(ValueError):
        load_and_validate_config(cfg)


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

