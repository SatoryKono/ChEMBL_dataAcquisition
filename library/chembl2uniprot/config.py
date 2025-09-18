"""Configuration loading and validation utilities using Pydantic.

The :func:`load_and_validate_config` function reads a YAML configuration file,
validates it against a JSON schema located next to the config file and builds a
:class:`Config` object with strong type checks. Environment variables prefixed
with ``CHEMBL_DA__`` or the legacy ``CHEMBL_`` override values from the YAML
file (e.g. ``CHEMBL_DA__CHEMBL2UNIPROT__BATCH__SIZE=5``).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, cast

import yaml
from jsonschema import Draft202012Validator
from pydantic import BaseModel, Field, ValidationError

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

    level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]
    format: Literal["human", "json"] = "human"


class Config(BaseModel):
    """Validated application configuration.

    This class defines the overall configuration structure by composing
    the other models. It represents the single source of truth for all
    configurable parameters in the application.

    Attributes
    ----------
    io:
        I/O related configuration.
    columns:
        Names of relevant CSV columns.
    uniprot:
        Configuration related to UniProt service access.
    network:
        Network level configuration.
    batch:
        Batch processing settings.
    logging:
        Logging configuration.
    """

    io: IOConfig
    columns: ColumnsConfig
    uniprot: UniprotConfig
    network: NetworkConfig
    batch: BatchConfig
    logging: LoggingConfig


@dataclass(frozen=True)
class ResolvedRuntimeOptions:
    """Effective runtime options derived from YAML and CLI inputs."""

    log_level: str
    log_format: Literal["human", "json"]
    separator: str
    input_encoding: str
    output_encoding: str


def resolve_runtime_options(
    cfg: Config,
    *,
    cli_log_level: str | None = None,
    cli_log_format: Literal["human", "json"] | None = None,
    cli_sep: str | None = None,
    cli_encoding: str | None = None,
) -> ResolvedRuntimeOptions:
    """Merge CLI overrides with the YAML configuration values.

    Parameters
    ----------
    cfg:
        Parsed configuration obtained from :func:`load_and_validate_config`.
    cli_log_level:
        Optional log level supplied via the command line interface.  When
        ``None`` the YAML value is used.
    cli_log_format:
        Optional log format specified via CLI.  ``None`` falls back to the
        configuration file.
    cli_sep:
        CSV column separator provided through CLI.  ``None`` preserves the
        separator from the configuration file.
    cli_encoding:
        Text encoding supplied by the CLI.  ``None`` keeps the input and output
        encodings defined in the configuration file.

    Returns
    -------
    ResolvedRuntimeOptions
        Object containing the effective runtime options for logging and CSV I/O.
    """

    log_level = cli_log_level if cli_log_level is not None else cfg.logging.level
    log_format = cli_log_format if cli_log_format is not None else cfg.logging.format
    separator = cli_sep if cli_sep is not None else cfg.io.csv.separator
    input_encoding = cli_encoding if cli_encoding is not None else cfg.io.input.encoding
    output_encoding = (
        cli_encoding if cli_encoding is not None else cfg.io.output.encoding
    )
    return ResolvedRuntimeOptions(
        log_level=log_level,
        log_format=log_format,
        separator=separator,
        input_encoding=input_encoding,
        output_encoding=output_encoding,
    )


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
    """Read a YAML file and return its content as a dictionary.

    Parameters
    ----------
    path:
        Path to the YAML file.

    Returns
    -------
    Dict[str, Any]
        The content of the YAML file.
    """
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return cast(Dict[str, Any], data)


def _read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file and return its content as a dictionary.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    Dict[str, Any]
        The content of the JSON file.
    """
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return cast(Dict[str, Any], data)


def _apply_env_overrides(
    data: Dict[str, Any], *, section: str | None = None
) -> Dict[str, Any]:
    """Update ``data`` with environment variable overrides.

    Both the project-scoped ``CHEMBL_DA__`` prefix and the legacy ``CHEMBL_``
    prefix are recognised. The remainder of the variable name is interpreted as
    a configuration path where components are separated by double underscores.
    For example, ``CHEMBL_DA__CHEMBL2UNIPROT__BATCH__SIZE=10`` overrides the
    ``batch.size`` key when the ``chembl2uniprot`` section is loaded.

    Parameters
    ----------
    data:
        The configuration dictionary to update.
    section:
        Optional name of the configuration section that is currently being
        processed. When provided the section prefix in the environment variable
        is ignored, allowing users to target keys via
        ``CHEMBL_DA__<SECTION>__...``.

    Returns
    -------
    Dict[str, Any]
        The updated configuration dictionary.
    """

    prefixes = ("CHEMBL_DA__", "CHEMBL_")
    section_lower = section.lower() if section else None
    for raw_key, value in os.environ.items():
        existing_keys = {str(key).lower() for key in data}
        matched_prefix = next((p for p in prefixes if raw_key.startswith(p)), None)
        if matched_prefix is None:
            continue
        tail = raw_key[len(matched_prefix) :]
        if not tail:
            continue
        path = [part.lower() for part in tail.split("__") if part]
        if not path:
            continue
        if matched_prefix == "CHEMBL_DA__":
            if section_lower and path[0] == section_lower:
                path = path[1:]
            elif not section_lower and path[0] not in existing_keys and len(path) > 1:
                path = path[1:]
        if not path:
            continue
        ref: Dict[str, Any] | Any = data
        valid_path = True
        for part in path[:-1]:
            if not isinstance(ref, dict):
                valid_path = False
                break
            ref = ref.setdefault(part, {})
        if not valid_path or not isinstance(ref, dict):
            continue
        ref[path[-1]] = value
    return data


def load_and_validate_config(
    config_path: str | Path,
    schema_path: str | Path | None = None,
    *,
    section: str | None = None,
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
    section:
        Optional top-level key within the YAML file.  When provided, only this
        subsection is validated against the schema.  Useful when multiple
        configurations share a single file.

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
    if section:
        try:
            config_dict = config_dict[section]
        except KeyError as exc:  # pragma: no cover - defensive programming
            raise KeyError(f"Section '{section}' not found in {config_path}") from exc
    columns_dict = _normalise_column_aliases(config_dict.get("columns", {}))
    config_dict["columns"] = columns_dict

    schema_dict = _read_json(schema_path)
    validator = Draft202012Validator(schema_dict)
    errors = sorted(validator.iter_errors(config_dict), key=lambda e: e.path)
    if errors:
        messages: list[str] = []
        for err in errors:
            path_parts = [str(p) for p in err.absolute_path]
            value: Any = err.instance
            if err.validator == "additionalProperties" and isinstance(
                err.instance, dict
            ):
                match = re.search(r"'(.+?)' was unexpected", err.message)
                if match:
                    key = match.group(1)
                    path_parts.append(key)
                    value = err.instance.get(key)
            path = ".".join(path_parts) or "<root>"
            LOGGER.error(
                "Configuration validation error at %s: %r - %s",
                path,
                value,
                err.message,
            )
            messages.append(f"{path}: {err.message}")
        raise ValueError("Configuration validation error(s): " + "; ".join(messages))

    _apply_env_overrides(config_dict, section=section)
    LOGGER.debug("Loaded configuration from %s", config_path)
    try:
        return Config(**config_dict)
    except ValidationError as exc:  # pragma: no cover - exercised in tests
        raise ValueError(str(exc)) from exc
