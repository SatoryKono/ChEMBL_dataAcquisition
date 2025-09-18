"""Configuration models for configurable scripts and utilities."""

from .uniprot import (
    ConfigError,
    HttpCacheConfig,
    OrthologsConfig,
    OutputConfig,
    UniProtScriptConfig,
    UniProtSection,
    load_uniprot_target_config,
)

__all__ = [
    "ConfigError",
    "HttpCacheConfig",
    "OrthologsConfig",
    "OutputConfig",
    "UniProtScriptConfig",
    "UniProtSection",
    "load_uniprot_target_config",
]
