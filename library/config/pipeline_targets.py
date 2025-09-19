"""Typed configuration models for :mod:`scripts.pipeline_targets_main`.

The pipeline targets CLI historically consumed loosely structured dictionaries
derived from YAML configuration files.  The new Pydantic models in this module
provide strong typing and validation for the sections consumed by
``pipeline_targets_main`` which reduces the amount of defensive code required
at runtime.  All models accept both the legacy flat key structure
(``timeout_sec``/``retries``/``rps`` directly under a section) and the new
nested layout using ``network`` and ``rate_limit`` sub-sections.  Invalid data
types or missing required keys result in immediate validation errors, making
configuration issues easier to diagnose.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    field_validator,
    model_validator,
)

from library.http_client import CacheConfig


def _normalise_network_dict(data: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a dictionary with canonical retry/backoff keys."""

    if data is None:
        return {}
    if not isinstance(data, Mapping):
        msg = "Network configuration must be a mapping"
        raise TypeError(msg)
    normalised = dict(data)
    alias_map = {
        "timeout": "timeout_sec",
        "timeout_seconds": "timeout_sec",
        "timeout_s": "timeout_sec",
        "retries": "max_retries",
        "retry": "max_retries",
        "max_retry_attempts": "max_retries",
        "backoff": "backoff_sec",
        "backoff_seconds": "backoff_sec",
        "backoff_base_sec": "backoff_sec",
        "backoff_base_seconds": "backoff_sec",
    }
    for alias, target in alias_map.items():
        if alias in normalised and target not in normalised:
            normalised[target] = normalised.pop(alias)
        elif alias in normalised:
            normalised.pop(alias)
    return normalised


def _normalise_rate_limit_dict(data: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a dictionary containing a canonical ``rps`` entry."""

    if data is None:
        return {}
    if not isinstance(data, Mapping):
        msg = "Rate limit configuration must be a mapping"
        raise TypeError(msg)
    normalised = dict(data)
    for alias in ("rate_limit_rps", "requests_per_second", "req_per_sec"):
        if alias in normalised and "rps" not in normalised:
            normalised["rps"] = normalised.pop(alias)
        elif alias in normalised:
            normalised.pop(alias)
    return normalised


def _normalise_section_payload(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``data`` with normalised ``network`` and ``rate_limit`` entries."""

    result = dict(data)

    # Extract legacy flat keys so they can be merged into the nested sections.
    network_aliases = {
        "timeout_sec": "timeout_sec",
        "timeout": "timeout_sec",
        "timeout_seconds": "timeout_sec",
        "retries": "max_retries",
        "max_retries": "max_retries",
        "backoff": "backoff_sec",
        "backoff_sec": "backoff_sec",
        "backoff_seconds": "backoff_sec",
        "backoff_base_sec": "backoff_sec",
    }
    legacy_network: dict[str, Any] = {}
    for alias, target in network_aliases.items():
        if alias in result:
            legacy_network[target] = result.pop(alias)

    rate_aliases = {
        "rps": "rps",
        "rate_limit_rps": "rps",
        "requests_per_second": "rps",
    }
    legacy_rate: dict[str, Any] = {}
    for alias, target in rate_aliases.items():
        if alias in result:
            legacy_rate[target] = result.pop(alias)

    network_payload = _normalise_network_dict(result.pop("network", None))
    for key, value in legacy_network.items():
        network_payload.setdefault(key, value)
    result["network"] = network_payload

    rate_payload = _normalise_rate_limit_dict(result.pop("rate_limit", None))
    for key, value in legacy_rate.items():
        rate_payload.setdefault(key, value)
    result["rate_limit"] = rate_payload

    cache_value = result.get("cache")
    if cache_value is not None:
        if not isinstance(cache_value, Mapping):
            msg = "Cache configuration must be a mapping"
            raise TypeError(msg)
        result["cache"] = dict(cache_value)

    return result


def _normalise_str_list(value: Any, *, field_name: str) -> list[str]:
    """Return ``value`` as a cleaned list of non-empty strings."""

    if value is None:
        return []
    if isinstance(value, str):
        candidates: Sequence[str] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        candidates = value
    else:
        msg = f"{field_name} must be a sequence of strings"
        raise ValueError(msg)
    cleaned: list[str] = []
    for entry in candidates:
        if not isinstance(entry, str):
            msg = f"{field_name} entries must be strings"
            raise ValueError(msg)
        text = entry.strip()
        if not text:
            msg = f"{field_name} entries must not be blank"
            raise ValueError(msg)
        cleaned.append(text)
    return cleaned


class CacheSettings(BaseModel):
    """HTTP cache configuration shared by individual sections."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    path: str | None = None
    ttl_sec: NonNegativeFloat = Field(default=0.0)

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str | None) -> str | None:
        """Ensure cache paths are not blank when provided."""

        if value is None:
            return None
        path_str = str(value).strip()
        if not path_str:
            msg = "cache.path must not be blank"
            raise ValueError(msg)
        return path_str

    def to_cache_config(self) -> CacheConfig | None:
        """Return a :class:`~library.http_client.CacheConfig` instance."""

        payload = {
            "enabled": self.enabled,
            "path": self.path,
            "ttl_sec": self.ttl_sec,
        }
        return CacheConfig.from_dict(payload)


class NetworkSettings(BaseModel):
    """Retry and timeout behaviour for HTTP clients."""

    model_config = ConfigDict(extra="forbid")

    timeout_sec: PositiveFloat | None = Field(default=None)
    max_retries: NonNegativeInt | None = Field(default=None)
    backoff_sec: NonNegativeFloat | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def _normalise_aliases(cls, data: Any) -> Any:
        """Accept legacy alternative keys such as ``retries`` or ``backoff``."""

        if data is None:
            return {}
        if isinstance(data, Mapping):
            return _normalise_network_dict(data)
        msg = "network must be a mapping"
        raise TypeError(msg)

    def effective_timeout(self, default: float) -> float:
        """Return ``timeout_sec`` falling back to ``default`` when unset."""

        return float(self.timeout_sec or default)

    def effective_retries(self, default: int) -> int:
        """Return ``max_retries`` falling back to ``default`` when unset."""

        return int(self.max_retries if self.max_retries is not None else default)

    def effective_backoff(self, default: float) -> float:
        """Return ``backoff_sec`` falling back to ``default`` when unset."""

        return float(self.backoff_sec if self.backoff_sec is not None else default)


class RateLimitSettings(BaseModel):
    """Rate limit configuration for HTTP clients."""

    model_config = ConfigDict(extra="forbid")

    rps: PositiveFloat | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def _normalise_aliases(cls, data: Any) -> Any:
        """Accept legacy aliases for the requests-per-second limit."""

        if data is None:
            return {}
        if isinstance(data, Mapping):
            return _normalise_rate_limit_dict(data)
        msg = "rate_limit must be a mapping"
        raise TypeError(msg)

    def effective_rps(self, default: float) -> float:
        """Return ``rps`` falling back to ``default`` when unset."""

        return float(self.rps or default)


class SectionBase(BaseModel):
    """Base class for configuration sections with shared HTTP settings."""

    model_config = ConfigDict(extra="forbid")

    network: NetworkSettings = Field(default_factory=NetworkSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    cache: CacheSettings | None = None

    @model_validator(mode="before")
    @classmethod
    def _prepare_payload(cls, data: Any) -> Any:
        """Normalise legacy keys before model validation."""

        if data is None:
            return {}
        if isinstance(data, Mapping):
            return _normalise_section_payload(data)
        msg = f"Expected mapping for {cls.__name__}"
        raise TypeError(msg)


class UniProtSectionConfig(SectionBase):
    """Configuration for the UniProt client used by the pipeline CLI."""

    base_url: HttpUrl
    include_isoforms: bool = False
    use_fasta_stream_for_isoform_ids: bool = True
    columns: list[str] = Field(default_factory=list)
    fields: list[str] | None = None

    @field_validator("columns", mode="before")
    @classmethod
    def _normalise_columns(cls, value: Any) -> list[str]:
        return _normalise_str_list(value, field_name="uniprot.columns")

    @field_validator("columns")
    @classmethod
    def _validate_columns(cls, value: list[str]) -> list[str]:
        return value

    @field_validator("fields", mode="before")
    @classmethod
    def _normalise_fields(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        return _normalise_str_list(value, field_name="uniprot.fields")


class HGNCServiceConfigModel(BaseModel):
    """Nested HGNC endpoint configuration."""

    model_config = ConfigDict(extra="forbid")

    base_url: HttpUrl


class HGNCOutputConfigModel(BaseModel):
    """Serialisation defaults for the HGNC CSV output."""

    model_config = ConfigDict(extra="forbid")

    sep: str = Field(default=",")
    encoding: str = Field(default="utf-8")

    @field_validator("sep")
    @classmethod
    def _validate_sep(cls, value: str) -> str:
        if value == "":
            msg = "hgnc.output.sep must not be empty"
            raise ValueError(msg)
        return value

    @field_validator("encoding")
    @classmethod
    def _validate_encoding(cls, value: str) -> str:
        if not value.strip():
            msg = "hgnc.output.encoding must not be blank"
            raise ValueError(msg)
        return value.strip()


class HGNCSectionConfig(SectionBase):
    """Configuration for the HGNC lookup client."""

    columns: list[str] = Field(default_factory=list)
    hgnc: HGNCServiceConfigModel
    network: NetworkSettings = Field(default_factory=NetworkSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    output: HGNCOutputConfigModel

    @field_validator("columns", mode="before")
    @classmethod
    def _normalise_columns(cls, value: Any) -> list[str]:
        return _normalise_str_list(value, field_name="hgnc.columns")


class GtoPSectionConfig(SectionBase):
    """Guide to PHARMACOLOGY (IUPHAR) client configuration."""

    base_url: HttpUrl
    columns: list[str] = Field(default_factory=list)

    @field_validator("columns", mode="before")
    @classmethod
    def _normalise_columns(cls, value: Any) -> list[str]:
        return _normalise_str_list(value, field_name="gtop.columns")


class OrthologsSectionConfig(SectionBase):
    """Configuration for ortholog enrichment clients (Ensembl and OMA)."""

    enabled: bool = True
    primary_source: str | None = None
    target_species: list[str] = Field(default_factory=list)
    species_priority: list[str] = Field(default_factory=list)

    @field_validator("primary_source")
    @classmethod
    def _validate_primary_source(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        if not text:
            msg = "orthologs.primary_source must not be blank"
            raise ValueError(msg)
        return text

    @field_validator("target_species", "species_priority", mode="before")
    @classmethod
    def _normalise_species_lists(cls, value: Any) -> list[str]:
        return _normalise_str_list(value, field_name="orthologs.species")


class PipelineClientsConfig(BaseModel):
    """Root configuration consumed by ``build_clients``."""

    model_config = ConfigDict(extra="allow")

    http_cache: CacheSettings | None = None
    uniprot: UniProtSectionConfig
    hgnc: HGNCSectionConfig
    gtop: GtoPSectionConfig
    orthologs: OrthologsSectionConfig | None = None
