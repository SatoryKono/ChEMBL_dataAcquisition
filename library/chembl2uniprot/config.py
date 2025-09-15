"""Configuration loading and validation utilities using Pydantic.

The :func:`load_and_validate_config` function reads a YAML configuration file,
validates it against a JSON schema located next to the config file and builds a
:class:`Config` object with strong type checks.  Environment variables prefixed
with ``CHEMBL_`` override values from the YAML file (e.g.
``CHEMBL_BATCH__SIZE=5``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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

    @model_validator(mode="before")
    @classmethod
    def _unify_chembl_id(cls, data: Any) -> Any:
        """Accept legacy ``target_chembl_id`` key as an alias for ``chembl_id``."""
        if isinstance(data, dict):
            if "target_chembl_id" in data and "chembl_id" not in data:
                data["chembl_id"] = data.pop("target_chembl_id")
        return data


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

    model_config = ConfigDict(extra="forbid")

    io: IOConfig
    columns: ColumnsConfig
    uniprot: UniprotConfig
    network: NetworkConfig
    batch: BatchConfig
    logging: LoggingConfig


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
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
    config_path: str | Path,
    schema_path: str | Path | None = None,
    *,
    section: str | None = None,
) -> Config:
    """Load ``config_path`` and validate it using Pydantic models.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.
    schema_path:
        DEPRECATED and unused.
    section:
        Optional top-level key within the YAML file.  When provided, only this
        subsection is loaded.

    Returns
    -------
    Config
        Parsed and validated configuration object.

    Raises
    ------
    ValueError
        If the configuration does not match the Pydantic models.
    FileNotFoundError
        If the configuration file cannot be found.
    """
    del schema_path  # Unused parameter

    config_path = Path(config_path)
    config_dict = _read_yaml(config_path)
    if section:
        try:
            config_dict = config_dict[section]
        except KeyError as exc:  # pragma: no cover - defensive programming
            raise KeyError(f"Section '{section}' not found in {config_path}") from exc

    _apply_env_overrides(config_dict)
    LOGGER.debug("Loaded configuration from %s", config_path)
    try:
        return Config(**config_dict)
    except ValidationError as exc:
        error_lines = [
            f"  - {'.'.join(str(i) for i in err['loc'])}: {err['msg']}"
            for err in exc.errors()
        ]
        LOGGER.error(
            "Configuration validation failed:\n%s",
            "\n".join(error_lines),
        )
        raise ValueError("Configuration validation failed") from exc
