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
class Config:
    """Dataclass wrapper around the configuration dictionary.

    The structure of the dataclass mirrors the validated JSON configuration.  It
    is provided mostly for type checking convenience.  Consumers typically use
    the :attr:`raw` attribute to access values.
    """

    raw: Dict[str, Any]


def _read_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a mapping at the root")
    return cast(Dict[str, Any], data)


def _read_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Schema file must contain a JSON object at the root")
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
    return Config(raw=config_dict)
