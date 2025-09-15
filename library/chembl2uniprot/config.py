"""Configuration loading and validation utilities.

The :func:`load_and_validate_config` function reads a YAML configuration file
and validates it against a JSON schema located next to the config file.  A
``ValueError`` with a meaningful message is raised when the configuration does
not conform to the schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, cast
import json
import logging

import yaml
from jsonschema import Draft202012Validator

LOGGER = logging.getLogger(__name__)


@dataclass
class EncodingConfig:
    """Encoding information for CSV input/output."""

    encoding: str


@dataclass
class CSVConfig:
    """CSV formatting details."""

    separator: str
    multivalue_delimiter: str


@dataclass
class IOConfig:
    """I/O related configuration."""

    input: EncodingConfig
    output: EncodingConfig
    csv: CSVConfig


@dataclass
class ColumnsConfig:
    """Names of relevant CSV columns."""

    chembl_id: str
    uniprot_out: str


@dataclass
class IdMappingConfig:
    """Endpoints and database names for UniProt ID mapping."""

    endpoint: str
    status_endpoint: str | None = None
    results_endpoint: str | None = None
    from_db: str | None = None
    to_db: str | None = None


@dataclass
class PollingConfig:
    """Polling behaviour for asynchronous jobs.

    Parameters
    ----------
    interval_sec:
        Delay between polling requests in seconds.
    max_polls:
        Maximum number of status requests before giving up. ``None`` disables the
        limit.
    total_timeout_sec:
        Total allowed time spent polling before aborting. ``None`` disables the
        limit.
    """

    interval_sec: float
    max_polls: int | None = None
    total_timeout_sec: float | None = None


@dataclass
class RateLimitConfig:
    """Rate limiting settings."""

    rps: float


@dataclass
class RetryConfig:
    """Retry configuration for HTTP requests."""

    max_attempts: int
    backoff_sec: float


@dataclass
class UniprotConfig:
    """Configuration related to UniProt service access."""

    base_url: str
    id_mapping: IdMappingConfig
    polling: PollingConfig
    rate_limit: RateLimitConfig
    retry: RetryConfig


@dataclass
class NetworkConfig:
    """Network level configuration."""

    timeout_sec: float


@dataclass
class BatchConfig:
    """Batch processing settings."""

    size: int


@dataclass
class LoggingConfig:
    """Logging level configuration."""

    level: str


@dataclass
class Config:
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


def _build_config(data: Dict[str, Any]) -> Config:
    """Construct a :class:`Config` object from ``data``.

    Parameters
    ----------
    data:
        Raw dictionary read from the YAML configuration file.  Assumes the
        structure matches the JSON schema.
    """
    # Accept legacy configuration where ``target_chembl_id`` was used instead
    # of ``chembl_id`` and standardise on the modern key.
    columns_cfg = _normalise_column_aliases(dict(data["columns"]), drop_legacy=True)

    io_cfg = IOConfig(
        input=EncodingConfig(**data["io"]["input"]),
        output=EncodingConfig(**data["io"]["output"]),
        csv=CSVConfig(**data["io"]["csv"]),
    )
    uniprot_cfg = UniprotConfig(
        base_url=data["uniprot"]["base_url"],
        id_mapping=IdMappingConfig(**data["uniprot"]["id_mapping"]),
        polling=PollingConfig(**data["uniprot"]["polling"]),
        rate_limit=RateLimitConfig(**data["uniprot"]["rate_limit"]),
        retry=RetryConfig(**data["uniprot"]["retry"]),
    )
    return Config(
        io=io_cfg,
        columns=ColumnsConfig(**columns_cfg),
        uniprot=uniprot_cfg,
        network=NetworkConfig(**data["network"]),
        batch=BatchConfig(**data["batch"]),
        logging=LoggingConfig(**data["logging"]),
    )


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return cast(Dict[str, Any], data)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return cast(Dict[str, Any], data)


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
        Dataclass encapsulating the validated configuration dictionary.

    Raises
    ------
    ValueError
        If the configuration does not match the schema.
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
            # Build a dotted path to the offending element for clarity.
            path = ".".join(str(p) for p in err.absolute_path)
            messages.append(f"{path or '<root>'}: {err.message}")
        raise ValueError("Configuration validation error(s): " + "; ".join(messages))

    LOGGER.debug("Loaded configuration from %s", config_path)
    return _build_config(config_dict)
