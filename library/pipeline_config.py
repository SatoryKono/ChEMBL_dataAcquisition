"""Pydantic models for pipeline-related configuration sections.

The utilities in this module provide strongly typed representations for the
``pipeline``, ``hgnc`` and ``orthologs`` sections found in the project's YAML
configuration files.  They are designed to validate configuration data at load
-time and surface helpful errors when values are missing or malformed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared defaults
# ---------------------------------------------------------------------------

DEFAULT_PIPELINE_COLUMNS: List[str] = [
    "target_chembl_id",
    "uniprot_id_primary",
    "uniprot_ids_all",
    "isoform_ids",
    "isoform_names",
    "isoform_synonyms",
    "hgnc_id",
    "gene_symbol",
    "protein_name_canonical",
    "protein_name_alt",
    "names_all",
    "synonyms_all",
    "organism",
    "taxon_id",
    "lineage_superkingdom",
    "lineage_phylum",
    "lineage_class",
    "lineage_order",
    "lineage_family",
    "target_type",
    "protein_class_L1",
    "protein_class_L2",
    "protein_class_L3",
    "protein_class_L4",
    "protein_class_L5",
    "sequence_length",
    "features_signal_peptide",
    "features_transmembrane",
    "features_topology",
    "ptm_glycosylation",
    "ptm_lipidation",
    "ptm_disulfide_bond",
    "ptm_modified_residue",
    "pfam",
    "interpro",
    "xref_chembl",
    "xref_uniprot",
    "xref_ensembl",
    "xref_pdb",
    "xref_alphafold",
    "xref_iuphar",
    "gtop_target_id",
    "gtop_synonyms",
    "ortholog_uniprot_ids",
    "orthologs_json",
    "orthologs_count",
    "gtop_natural_ligands_n",
    "gtop_interactions_n",
    "gtop_function_text_short",
    "uniprot_last_update",
    "uniprot_version",
    "pipeline_version",
    "timestamp_utc",
]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class IupharSettings(BaseModel):
    """Configuration for the optional IUPHAR/GtoP enrichment step."""

    model_config = ConfigDict(extra="forbid")

    affinity_parameter: str = "pKi"
    approved_only: Optional[bool] = None
    primary_target_only: bool = True


class PipelineSettings(BaseModel):
    """Settings controlling the unified target data pipeline."""

    model_config = ConfigDict(extra="forbid")

    rate_limit_rps: float = Field(default=2.0, gt=0)
    retries: int = Field(default=3, ge=0)
    timeout_sec: float = Field(default=30.0, gt=0)
    species_priority: List[str] = Field(
        default_factory=lambda: ["Human", "Homo sapiens"]
    )
    list_format: str = Field(default="json", min_length=1)
    include_sequence: bool = False
    include_isoforms: bool = False
    columns: List[str] = Field(default_factory=list)
    iuphar: IupharSettings = Field(default_factory=IupharSettings)


class HgncServiceSettings(BaseModel):
    """Network endpoint configuration for HGNC lookups."""

    model_config = ConfigDict(extra="forbid")

    base_url: str


class HgncNetworkSettings(BaseModel):
    """Retry and timeout configuration for HGNC HTTP requests."""

    model_config = ConfigDict(extra="forbid")

    timeout_sec: float = Field(gt=0)
    max_retries: int = Field(ge=0)


class HgncRateLimitSettings(BaseModel):
    """Rate limiting applied to HGNC API calls."""

    model_config = ConfigDict(extra="forbid")

    rps: float = Field(gt=0)


class HgncOutputSettings(BaseModel):
    """Output formatting configuration for HGNC CSV exports."""

    model_config = ConfigDict(extra="forbid")

    sep: str = Field(min_length=1)
    encoding: str = Field(min_length=1)


class HgncSettings(BaseModel):
    """Aggregate HGNC configuration section."""

    model_config = ConfigDict(extra="forbid")

    columns: List[str] = Field(default_factory=list)
    hgnc: HgncServiceSettings
    network: HgncNetworkSettings
    rate_limit: HgncRateLimitSettings
    output: HgncOutputSettings


class OrthologsSettings(BaseModel):
    """Configuration for ortholog retrieval from Ensembl/OMA services."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    primary_source: Optional[str] = None
    target_species: List[str] = Field(default_factory=list)
    species_priority: List[str] = Field(default_factory=list)
    rate_limit_rps: float = Field(default=2.0, gt=0)
    timeout_sec: float = Field(default=30.0, gt=0)
    retries: int = Field(default=3, ge=0)
    backoff_base_sec: float = Field(default=1.0, ge=0)


# ---------------------------------------------------------------------------
# Loader utilities
# ---------------------------------------------------------------------------


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    """Read ``path`` as UTF-8 encoded YAML and return the resulting mapping."""

    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):  # pragma: no cover - defensive validation
        raise TypeError(f"Configuration at {path} must be a mapping")
    return cast(Dict[str, Any], raw)


def load_pipeline_settings(path: str | Path) -> PipelineSettings:
    """Load and validate the ``pipeline`` configuration section.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    PipelineSettings
        Validated pipeline configuration. Missing keys fall back to sensible
        defaults defined by :class:`PipelineSettings`.
    """

    data = _read_yaml(path)
    section = data.get("pipeline", {})
    return PipelineSettings.model_validate(section)


def load_hgnc_settings(
    path: str | Path, *, section: str | None = "hgnc"
) -> HgncSettings:
    """Load and validate HGNC configuration settings.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.
    section:
        Optional key selecting a nested dictionary within ``path``.  When set to
        ``None`` the entire file content is interpreted as the HGNC section.

    Returns
    -------
    HgncSettings
        Validated HGNC configuration.

    Raises
    ------
    KeyError
        If ``section`` is provided and not present in the YAML document.
    """

    data = _read_yaml(path)
    if section is not None:
        try:
            data = data[section]
        except KeyError as exc:  # pragma: no cover - defensive validation
            raise KeyError(f"Section '{section}' not found in {path}") from exc
    return HgncSettings.model_validate(data)


def load_orthologs_settings(path: str | Path) -> OrthologsSettings:
    """Load and validate the ``orthologs`` configuration section.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    OrthologsSettings
        Validated configuration. When the ``orthologs`` section is missing a
        model populated with default values is returned.
    """

    data = _read_yaml(path)
    section = data.get("orthologs", {})
    return OrthologsSettings.model_validate(section)


__all__ = [
    "DEFAULT_PIPELINE_COLUMNS",
    "IupharSettings",
    "PipelineSettings",
    "HgncServiceSettings",
    "HgncNetworkSettings",
    "HgncRateLimitSettings",
    "HgncOutputSettings",
    "HgncSettings",
    "OrthologsSettings",
    "load_pipeline_settings",
    "load_hgnc_settings",
    "load_orthologs_settings",
]
