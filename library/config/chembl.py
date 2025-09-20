"""Configuration models for ChEMBL-centric CLI scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

T = TypeVar("T", bound="ChemblScriptConfig")


def _normalise_section_payload(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``data`` with nested ``csv``/``network``/``rate_limit`` payloads."""

    result = dict(data)

    csv_payload: dict[str, Any] = {}
    existing_csv = result.pop("csv", None)
    if isinstance(existing_csv, Mapping):
        csv_payload.update(existing_csv)
    for alias, target in (
        ("separator", "sep"),
        ("sep", "sep"),
        ("encoding", "encoding"),
        ("list_format", "list_format"),
    ):
        if alias in result and target not in csv_payload:
            csv_payload[target] = result.pop(alias)
        elif alias in result:
            result.pop(alias)
    if csv_payload:
        result["csv"] = csv_payload

    network_payload: dict[str, Any] = {}
    existing_network = result.pop("network", None)
    if isinstance(existing_network, Mapping):
        network_payload.update(existing_network)
    network_aliases = {
        "timeout": "timeout_sec",
        "timeout_s": "timeout_sec",
        "timeout_seconds": "timeout_sec",
        "timeout_sec": "timeout_sec",
        "retries": "max_retries",
        "max_retry_attempts": "max_retries",
        "retry": "max_retries",
        "max_retries": "max_retries",
        "retry_penalty": "retry_penalty_sec",
        "retry_penalty_seconds": "retry_penalty_sec",
        "retry_penalty_sec": "retry_penalty_sec",
        "penalty_sec": "retry_penalty_sec",
    }
    for alias, target in network_aliases.items():
        if alias in result and target not in network_payload:
            network_payload[target] = result.pop(alias)
        elif alias in result:
            result.pop(alias)
    if network_payload:
        result["network"] = network_payload

    rate_payload: dict[str, Any] = {}
    existing_rate = result.pop("rate_limit", None)
    if isinstance(existing_rate, Mapping):
        rate_payload.update(existing_rate)
    for alias in ("rps", "rate_limit_rps", "requests_per_second"):
        if alias in result and "rps" not in rate_payload:
            rate_payload["rps"] = result.pop(alias)
        elif alias in result:
            result.pop(alias)
    if rate_payload:
        result["rate_limit"] = rate_payload

    return result


class ChemblCsvConfig(BaseModel):
    """Serialisation options shared by the CLI entry points."""

    model_config = ConfigDict(extra="forbid")

    sep: str = Field(default=",")
    encoding: str = Field(default="utf-8")
    list_format: str = Field(default="json")

    @field_validator("sep")
    @classmethod
    def _non_empty_separator(cls, value: str) -> str:
        """Ensure ``sep`` is a non-empty string."""

        if value == "":
            msg = "Separator must not be empty"
            raise ValueError(msg)
        return value

    @field_validator("encoding")
    @classmethod
    def _validate_encoding(cls, value: str) -> str:
        """Normalise whitespace-only encodings to errors."""

        cleaned = value.strip()
        if not cleaned:
            msg = "Encoding must not be blank"
            raise ValueError(msg)
        return cleaned

    @field_validator("list_format")
    @classmethod
    def _validate_list_format(cls, value: str) -> str:
        """Restrict ``list_format`` to supported serialisation strategies."""

        allowed = {"json", "pipe"}
        if value not in allowed:
            msg = f"list_format must be one of {sorted(allowed)}"
            raise ValueError(msg)
        return value


class ChemblNetworkConfig(BaseModel):
    """Timeout and retry behaviour for ChEMBL API clients."""

    model_config = ConfigDict(extra="forbid")

    timeout_sec: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    retry_penalty_sec: float = Field(default=1.0, ge=0)


class ChemblRateLimitConfig(BaseModel):
    """Rate limiting configuration for the ChEMBL API client."""

    model_config = ConfigDict(extra="forbid")

    rps: float = Field(default=2.0, gt=0)


class ChemblScriptConfig(BaseModel):
    """Common configuration shared by ChEMBL assay/activity CLIs."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(default="https://www.ebi.ac.uk/chembl/api/data")
    user_agent: str = Field(default="ChEMBLDataAcquisition/1.0")
    chunk_size: int = Field(default=20, gt=0)
    csv: ChemblCsvConfig = Field(default_factory=ChemblCsvConfig)
    network: ChemblNetworkConfig = Field(default_factory=ChemblNetworkConfig)
    rate_limit: ChemblRateLimitConfig = Field(default_factory=ChemblRateLimitConfig)

    @model_validator(mode="before")
    @classmethod
    def _merge_aliases(cls, data: Any) -> Any:
        """Support legacy flat keys for nested configuration blocks."""

        if isinstance(data, Mapping):
            return _normalise_section_payload(data)
        return data

    @field_validator("base_url", "user_agent")
    @classmethod
    def _validate_non_blank(cls, value: str) -> str:
        """Ensure textual fields are non-empty after stripping whitespace."""

        cleaned = value.strip()
        if not cleaned:
            msg = "Value must not be blank"
            raise ValueError(msg)
        return cleaned


class ChemblActivitiesConfig(ChemblScriptConfig):
    """Configuration model for ``chembl_activities_main.py``."""


class ChemblAssaysConfig(ChemblScriptConfig):
    """Configuration model for ``chembl_assays_main.py``."""


def _load_section_config(
    path: str | Path,
    section: str,
    model: type[T],
    *,
    apply_env: bool = True,
) -> T:
    """Load ``section`` from ``path`` and validate it using ``model``."""

    config_path = Path(path)
    try:
        content = config_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise

    try:
        loaded = yaml.safe_load(content) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - invalid YAML handled elsewhere
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(loaded, Mapping):
        msg = f"Configuration root must be a mapping in {config_path}"
        raise ValueError(msg)

    payload = dict(loaded)

    if apply_env:
        try:
            from library.chembl2uniprot.config import _apply_env_overrides
        except Exception:  # pragma: no cover - defensive fallback
            pass
        else:
            _apply_env_overrides(payload, section=section)

    section_payload = payload.get(section)
    if section_payload is None:
        msg = f"Missing '{section}' section in {config_path}"
        raise ValueError(msg)

    try:
        return model.model_validate(section_payload)
    except ValidationError as exc:  # pragma: no cover - validation relayed to caller
        raise ValueError(str(exc)) from exc


def load_chembl_activities_config(
    path: str | Path, *, apply_env: bool = True
) -> ChemblActivitiesConfig:
    """Return the validated ``chembl_activities`` section from ``path``."""

    return _load_section_config(
        path,
        "chembl_activities",
        ChemblActivitiesConfig,
        apply_env=apply_env,
    )


def load_chembl_assays_config(
    path: str | Path, *, apply_env: bool = True
) -> ChemblAssaysConfig:
    """Return the validated ``chembl_assays`` section from ``path``."""

    return _load_section_config(
        path,
        "chembl_assays",
        ChemblAssaysConfig,
        apply_env=apply_env,
    )


__all__ = [
    "ChemblActivitiesConfig",
    "ChemblAssaysConfig",
    "ChemblCsvConfig",
    "ChemblNetworkConfig",
    "ChemblRateLimitConfig",
    "ChemblScriptConfig",
    "load_chembl_activities_config",
    "load_chembl_assays_config",
]
