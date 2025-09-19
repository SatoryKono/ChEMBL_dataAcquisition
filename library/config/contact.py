"""Contact and user agent configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ContactConfig(BaseModel):
    """Contact details and HTTP user agent configuration.

    Attributes
    ----------
    name:
        Human readable name for the maintainer or organisation operating the
        tooling. The value is used purely for documentation purposes but must
        not be blank.
    email:
        Optional contact email address. When provided it must include a single
        ``@`` separator and a domain.
    user_agent:
        Descriptive ``User-Agent`` header sent with outgoing HTTP requests.
        The string should generally include the project name and a contact
        point, for example ``"chembl-da/1.0 (mailto:team@example.org)"``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    email: str | None = None
    user_agent: str = Field(min_length=1)

    @field_validator("name", "user_agent")
    @classmethod
    def _strip_and_validate(cls, value: str) -> str:
        """Return ``value`` stripped of surrounding whitespace."""

        cleaned = value.strip()
        if not cleaned:
            msg = "Value must not be blank"
            raise ValueError(msg)
        return cleaned

    @field_validator("email")
    @classmethod
    def _validate_email(cls, value: str | None) -> str | None:
        """Ensure the optional email address is well formed."""

        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            msg = "Email address must not be blank"
            raise ValueError(msg)
        if cleaned.count("@") != 1:
            msg = "Email address must contain exactly one '@'"
            raise ValueError(msg)
        local_part, domain = cleaned.split("@", 1)
        if not local_part or not domain or "." not in domain:
            msg = "Email address must include a domain"
            raise ValueError(msg)
        return cleaned

    def effective_user_agent(self) -> str:
        """Return the user agent string guaranteed to be non-empty."""

        return self.user_agent


def load_contact_config(path: str | Path, *, apply_env: bool = True) -> ContactConfig:
    """Load and validate the ``contact`` section from a YAML configuration file.

    Parameters
    ----------
    path:
        Path to a YAML configuration file containing a top-level ``contact``
        mapping.
    apply_env:
        When ``True`` environment variable overrides following the
        ``CHEMBL_DA__`` convention are applied before validation.

    Returns
    -------
    ContactConfig
        Parsed and validated contact configuration.

    Raises
    ------
    FileNotFoundError
        If the provided path does not exist.
    ValueError
        If the YAML file does not contain a mapping or lacks the ``contact``
        section, or when validation fails.
    """

    config_path = Path(path)
    try:
        content = config_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise

    try:
        loaded = yaml.safe_load(content) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - invalid YAML exercised elsewhere
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(loaded, Mapping):
        msg = f"Configuration root must be a mapping in {config_path}"
        raise ValueError(msg)

    data = dict(loaded)

    if apply_env:
        try:
            from library.chembl2uniprot.config import _apply_env_overrides
        except Exception:  # pragma: no cover - defensive fallback
            pass
        else:
            _apply_env_overrides(data)

    contact_data: Any = data.get("contact")
    if contact_data is None:
        msg = f"Missing 'contact' section in {config_path}"
        raise ValueError(msg)

    try:
        return ContactConfig.model_validate(contact_data)
    except (
        ValidationError
    ) as exc:  # pragma: no cover - detailed error surfaced to caller
        raise ValueError(str(exc)) from exc
