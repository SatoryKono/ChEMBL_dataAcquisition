"""Configuration loading and validation utilities using Pydantic.

The :func:`load_and_validate_config` function reads a YAML configuration file,
validates it against a JSON schema located next to the config file and builds a
:class:`Config` object with strong type checks.  Environment variables prefixed
with ``CHEMBL_`` override values from the YAML file (e.g.
``CHEMBL_BATCH__SIZE=5``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Literal, cast

import yaml
from jsonschema import Draft202012Validator
from pydantic import BaseModel, Field, ValidationError
import os

LOGGER = logging.getLogger(__name__)


class EncodingConfig(BaseModel):
    """Encoding information for CSV input/output."""

    encoding: str


class CSVConfig(BaseModel):
    """CSV formatting details."""

    separator: str
    multivalue_delimiter: str


class IOConfig(BaseModel):
    """I/O related configuration."""

    input: EncodingConfig
    output: EncodingConfig
    csv: CSVConfig


class ColumnsConfig(BaseModel):
    """Names of relevant CSV columns."""

    chembl_id: str
    uniprot_out: str


class IdMappingConfig(BaseModel):
    """Endpoints and database names for UniProt ID mapping."""

    endpoint: str
    status_endpoint: str | None = None
    results_endpoint: str | None = None
    from_db: str | None = None
    to_db: str | None = None


class PollingConfig(BaseModel):
    """Polling behaviour for asynchronous jobs."""

    interval_sec: float = Field(ge=0)


class RateLimitConfig(BaseModel):
    """Rate limiting settings."""

    rps: float = Field(gt=0)


class RetryConfig(BaseModel):
    """Retry configuration for HTTP requests."""

    max_attempts: int = Field(gt=0)
    backoff_sec: float = Field(ge=0)


class UniprotConfig(BaseModel):
    """Configuration related to UniProt service access."""

    base_url: str
    id_mapping: IdMappingConfig
    polling: PollingConfig
    rate_limit: RateLimitConfig
    retry: RetryConfig


class NetworkConfig(BaseModel):
    """Network level configuration."""

    timeout_sec: float = Field(gt=0)


class BatchConfig(BaseModel):
    """Batch processing settings."""

    size: int = Field(gt=0)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str
    format: Literal["human", "json"] = "human"


class Config(BaseModel):
    """Validated application configuration."""

    io: IOConfig
    columns: ColumnsConfig
    uniprot: UniprotConfig
    network: NetworkConfig
    batch: BatchConfig
    logging: LoggingConfig


def _normalise_column_aliases(
    columns: Dict[str, Any], drop_legacy: bool = False
) -> Dict[str, Any]:
    """Ensure ``columns`` contains both old and new ChEMBL ID keys.

    Parameters
    ----------
    columns:
        Mapping of column names from the configuration file.
    drop_legacy:
        When ``True`` remove the legacy ``target_chembl_id`` key after
        normalisation.
    """

    if "chembl_id" in columns and "target_chembl_id" not in columns:
        columns["target_chembl_id"] = columns["chembl_id"]
    if "target_chembl_id" in columns and "chembl_id" not in columns:
        columns["chembl_id"] = columns["target_chembl_id"]
    if drop_legacy:
        columns.pop("target_chembl_id", None)
    return columns


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return cast(Dict[str, Any], data)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return cast(Dict[str, Any], data)


def _apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
    """Update ``data`` with ``CHEMBL_`` environment variable overrides."""

    prefix = "CHEMBL_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix) :].lower().split("__")
        ref = data
        for part in path[:-1]:
            ref = ref.setdefault(part, {})
        ref[path[-1]] = value
    return data


def load_and_validate_config(
    config_path: str | Path, schema_path: str | Path | None = None
) -> Config:
    """Load ``config_path`` and validate it against ``config.schema.json``.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.
    schema_path:
        Optional path to the JSON schema.  When ``None`` the schema is assumed
        to be located in the same directory as ``config_path`` under the name
        ``config.schema.json``.

    Returns
    -------
    Config
        Parsed and validated configuration object.

    Raises
    ------
    ValueError
        If the configuration does not match the schema or fails type checks.
    FileNotFoundError
        If either the configuration file or the schema file cannot be found.
    """

    config_path = Path(config_path)
    if schema_path is None:
        schema_path = config_path.with_name("config.schema.json")
    schema_path = Path(schema_path)

    config_dict = _read_yaml(config_path)
    columns_dict = _normalise_column_aliases(config_dict.get("columns", {}))
    config_dict["columns"] = columns_dict

    schema_dict = _read_json(schema_path)
    validator = Draft202012Validator(schema_dict)
    errors = sorted(validator.iter_errors(config_dict), key=lambda e: e.path)
    if errors:
        messages = []
        for err in errors:
            path = ".".join(str(p) for p in err.absolute_path)
            messages.append(f"{path or '<root>'}: {err.message}")
        raise ValueError("Configuration validation error(s): " + "; ".join(messages))

    _apply_env_overrides(config_dict)
    LOGGER.debug("Loaded configuration from %s", config_path)
    try:
        return Config(**config_dict)
    except ValidationError as exc:  # pragma: no cover - exercised in tests
        raise ValueError(str(exc)) from exc
