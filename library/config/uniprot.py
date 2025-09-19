"""Pydantic models for ``get_uniprot_target_data.py`` configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from library.chembl2uniprot.config import _apply_env_overrides


class ConfigError(ValueError):
    """Raised when a configuration file is syntactically valid but semantically invalid."""


class CacheSettings(BaseModel):
    """Shared cache configuration used by UniProt and ortholog clients."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    path: str | None = None
    ttl_sec: float = Field(default=0.0, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _normalise_ttl(cls, data: Any) -> Any:
        """Accept alternative TTL keys such as ``ttl`` or ``ttl_seconds``."""

        if isinstance(data, Mapping):
            if "ttl_sec" in data:
                return data
            normalised = dict(data)
            for alias in ("ttl", "ttl_seconds", "expire_after"):
                if alias in normalised:
                    normalised["ttl_sec"] = normalised[alias]
                    break
            return normalised
        return data

    @field_validator("path")
    @classmethod
    def _non_empty_path(cls, value: str | None) -> str | None:
        """Ensure the cache path is not empty when provided."""

        if value is None:
            return None
        if not str(value).strip():
            msg = "Cache path must not be blank"
            raise ValueError(msg)
        return str(value)

    def to_cache_dict(self) -> dict[str, Any]:
        """Return a mapping suitable for :class:`library.http_client.CacheConfig`."""

        data = self.model_dump()
        return {
            "enabled": data["enabled"],
            "path": data["path"],
            "ttl_sec": data["ttl_sec"],
        }


class OutputConfig(BaseModel):
    """Serialisation and formatting options for generated CSV files."""

    model_config = ConfigDict(extra="forbid")

    sep: str = Field(default=",")
    encoding: str = Field(default="utf-8")
    list_format: str = Field(default="json")
    include_sequence: bool = False

    @field_validator("sep")
    @classmethod
    def _validate_separator(cls, value: str) -> str:
        """Allow any non-empty separator, including whitespace characters."""

        if value == "":
            msg = "Separator must not be empty"
            raise ValueError(msg)
        return value

    @field_validator("encoding")
    @classmethod
    def _validate_encoding(cls, value: str) -> str:
        """Ensure the encoding hint is a non-blank string."""

        if not value or not value.strip():
            msg = "Encoding must not be blank"
            raise ValueError(msg)
        return value.strip()

    @field_validator("list_format")
    @classmethod
    def _validate_list_format(cls, value: str) -> str:
        """Ensure ``list_format`` is one of the supported encodings."""

        allowed = {"json", "pipe"}
        if value not in allowed:
            msg = f"list_format must be one of {sorted(allowed)}"
            raise ValueError(msg)
        return value


class CacheAwareSection(BaseModel):
    """Base model for sections supporting an optional nested cache configuration."""

    model_config = ConfigDict(extra="forbid")

    cache: CacheSettings | None = None


class UniProtSection(CacheAwareSection):
    """UniProt API access configuration."""

    base_url: str = Field(default="https://rest.uniprot.org/uniprotkb")
    include_isoforms: bool = False
    use_fasta_stream_for_isoform_ids: bool = True
    timeout_sec: float = Field(default=30.0, gt=0)
    retries: int = Field(default=3, ge=0)
    rps: float = Field(default=3.0, gt=0)
    batch_size: int = Field(default=100, gt=0)
    columns: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _merge_network_settings(cls, data: Any) -> Any:
        """Merge nested ``network``/``rate_limit`` subsections when provided."""

        if isinstance(data, Mapping):
            payload = dict(data)
            network = payload.pop("network", None)
            if isinstance(network, Mapping):
                payload.setdefault("timeout_sec", network.get("timeout_sec"))
                payload.setdefault("retries", network.get("max_retries"))
            rate_limit = payload.pop("rate_limit", None)
            if isinstance(rate_limit, Mapping):
                payload.setdefault("rps", rate_limit.get("rps"))
            return payload
        return data

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        """Require the UniProt base URL to be a non-empty string."""

        if not value or not value.strip():
            msg = "base_url must not be blank"
            raise ValueError(msg)
        return value

    @field_validator("columns", mode="before")
    @classmethod
    def _ensure_list(cls, value: Any) -> list[str]:
        """Accept any iterable of column names and normalise to ``list[str]``."""

        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, (set, tuple)):
            return list(value)
        if isinstance(value, str):
            return [value]
        msg = "columns must be a list of strings"
        raise TypeError(msg)

    @field_validator("columns")
    @classmethod
    def _validate_columns(cls, value: list[str]) -> list[str]:
        """Ensure column identifiers are non-empty strings."""

        cleaned: list[str] = []
        for column in value:
            column_str = str(column).strip()
            if not column_str:
                msg = "Column names must not be blank"
                raise ValueError(msg)
            cleaned.append(column_str)
        return cleaned


class OrthologsConfig(CacheAwareSection):
    """Configuration options for ortholog enrichment."""

    enabled: bool = True
    primary_source: str | None = None
    target_species: list[str] = Field(default_factory=list)
    species_priority: list[str] = Field(default_factory=list)
    rate_limit_rps: float = Field(default=2.0, gt=0)
    timeout_sec: float = Field(default=30.0, gt=0)
    retries: int = Field(default=3, ge=0)
    backoff_base_sec: float = Field(default=1.0, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _merge_network_settings(cls, data: Any) -> Any:
        """Support nested ``network`` and ``rate_limit`` subsections."""

        if isinstance(data, Mapping):
            payload = dict(data)
            network = payload.pop("network", None)
            if isinstance(network, Mapping):
                payload.setdefault("timeout_sec", network.get("timeout_sec"))
                payload.setdefault("retries", network.get("max_retries"))
                payload.setdefault("backoff_base_sec", network.get("backoff_sec"))
            rate_limit = payload.pop("rate_limit", None)
            if isinstance(rate_limit, Mapping):
                payload.setdefault("rate_limit_rps", rate_limit.get("rps"))
            return payload
        return data

    @field_validator("primary_source")
    @classmethod
    def _validate_primary_source(cls, value: str | None) -> str | None:
        """Trim whitespace and reject empty primary source identifiers."""

        if value is None:
            return None
        trimmed = value.strip()
        if not trimmed:
            msg = "primary_source must not be blank"
            raise ValueError(msg)
        return trimmed

    @field_validator("target_species", "species_priority", mode="before")
    @classmethod
    def _normalise_species(cls, value: Any) -> list[str]:
        """Normalise species definitions to ``list[str]``."""

        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, (set, tuple)):
            return list(value)
        if isinstance(value, str):
            return [value]
        msg = "Species lists must be sequences of strings"
        raise TypeError(msg)

    @field_validator("target_species", "species_priority")
    @classmethod
    def _validate_species_entries(cls, value: list[str]) -> list[str]:
        """Ensure species identifiers are non-empty strings."""

        result: list[str] = []
        for item in value:
            item_str = str(item).strip()
            if not item_str:
                msg = "Species names must not be blank"
                raise ValueError(msg)
            result.append(item_str)
        return result


class HttpCacheConfig(CacheSettings):
    """Top-level cache configuration shared across API clients."""


class UniProtScriptConfig(BaseModel):
    """Root configuration consumed by ``get_uniprot_target_data.py``."""

    model_config = ConfigDict(extra="ignore")

    output: OutputConfig = Field(default_factory=OutputConfig)
    uniprot: UniProtSection = Field(default_factory=UniProtSection)
    orthologs: OrthologsConfig = Field(default_factory=OrthologsConfig)
    http_cache: HttpCacheConfig | None = None


def _format_validation_error(path: Path, error: ValidationError) -> str:
    """Return a concise, user-friendly validation error message."""

    parts: list[str] = []
    for entry in error.errors():
        location = ".".join(str(component) for component in entry["loc"])
        parts.append(f"{location}: {entry['msg']}")
    details = "; ".join(parts)
    return f"Invalid configuration in {path}: {details}"


def load_uniprot_target_config(config_path: str | Path) -> UniProtScriptConfig:
    """Load and validate configuration for ``get_uniprot_target_data.py``.

    Parameters
    ----------
    config_path:
        Location of the YAML file providing the configuration sections.

    Returns
    -------
    UniProtScriptConfig
        Structured configuration describing output formatting, UniProt access,
        ortholog enrichment and HTTP caching defaults.

    Raises
    ------
    ConfigError
        Raised when the configuration file exists but does not match the
        expected schema.
    FileNotFoundError
        Propagated when ``config_path`` cannot be read.
    """

    path = Path(config_path)
    raw_text = path.read_text()
    try:
        loaded = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - rare syntax errors
        msg = f"Failed to parse YAML in {path}: {exc}"
        raise ConfigError(msg) from exc
    if not isinstance(loaded, Mapping):
        msg = f"Root of {path} must be a mapping"
        raise ConfigError(msg)

    sections: dict[str, Any] = {}
    for key in ("output", "uniprot", "orthologs", "http_cache"):
        value = loaded.get(key)
        if value is None:
            sections[key] = {}
            continue
        if not isinstance(value, Mapping):
            msg = f"Section '{key}' in {path} must be a mapping"
            raise ConfigError(msg)
        sections[key] = dict(value)

    _apply_env_overrides(sections)

    try:
        return UniProtScriptConfig.model_validate(sections)
    except ValidationError as error:  # pragma: no cover - exercised via tests
        message = _format_validation_error(path, error)
        raise ConfigError(message) from error
