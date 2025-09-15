# ruff: noqa: E402

"""CLI entry point for the unified target data pipeline."""

from __future__ import annotations

import argparse
import logging

import sys
from pathlib import Path

from typing import List

import pandas as pd
import yaml  # type: ignore[import]


ROOT = Path(__file__).resolve().parents[1]
LIB_DIR = ROOT / "library"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))


from chembl_targets import TargetConfig, fetch_targets
from gtop_client import GtoPClient, GtoPConfig
from hgnc_client import HGNCClient, load_config as load_hgnc_config
from uniprot_client import (
    NetworkConfig as UniNetworkConfig,
    RateLimitConfig as UniRateConfig,
    UniProtClient,
)
from orthologs import EnsemblHomologyClient, OmaClient


from pipeline_targets import PipelineConfig, load_pipeline_config, run_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline."""

    parser = argparse.ArgumentParser(description="Unified target data pipeline")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--id-column", default="target_chembl_id")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--sep", default=",")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--list-format", default="json")
    parser.add_argument("--species", default="Human")
    parser.add_argument("--affinity-parameter", default="pKi")
    parser.add_argument("--approved-only", default="false")
    parser.add_argument("--primary-target-only", default="true")
    parser.add_argument("--with-isoforms", action="store_true")
    parser.add_argument("--with-orthologs", action="store_true")
    return parser.parse_args()


def build_clients(
    cfg_path: str, pipeline_cfg: PipelineConfig, *, with_orthologs: bool = False
) -> tuple[
    UniProtClient,
    HGNCClient,
    GtoPClient,
    EnsemblHomologyClient | None,
    OmaClient | None,
    list[str],
]:
    """Initialise service clients used by the pipeline."""

    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    uni_cfg = data["uniprot"]
    fields = ",".join(uni_cfg.get("fields", []))
    uni = UniProtClient(
        base_url=uni_cfg["base_url"],
        fields=fields,
        network=UniNetworkConfig(
            timeout_sec=pipeline_cfg.timeout_sec,
            max_retries=pipeline_cfg.retries,
        ),
        rate_limit=UniRateConfig(rps=pipeline_cfg.rate_limit_rps),
    )

    # The HGNC configuration is nested under the top-level "hgnc" section
    # in the YAML file. Explicitly select this section to avoid passing the
    # entire configuration dictionary to ``HGNCServiceConfig``, which would
    # otherwise raise ``TypeError`` due to unexpected keys.
    hcfg = load_hgnc_config(cfg_path, section="hgnc")
    hcfg.network.timeout_sec = pipeline_cfg.timeout_sec
    hcfg.network.max_retries = pipeline_cfg.retries
    hcfg.rate_limit.rps = pipeline_cfg.rate_limit_rps
    hgnc = HGNCClient(hcfg)

    gcfg = GtoPConfig(
        base_url=data["gtop"]["base_url"],
        timeout_sec=pipeline_cfg.timeout_sec,
        max_retries=pipeline_cfg.retries,
        rps=pipeline_cfg.rate_limit_rps,
    )
    gtop = GtoPClient(gcfg)

    ens_client: EnsemblHomologyClient | None = None
    oma_client: OmaClient | None = None
    target_species: list[str] = []
    if with_orthologs:
        orth_cfg = data.get("orthologs", {})
        ens_client = EnsemblHomologyClient(
            base_url="https://rest.ensembl.org",
            network=UniNetworkConfig(
                timeout_sec=pipeline_cfg.timeout_sec,
                max_retries=pipeline_cfg.retries,
            ),
            rate_limit=UniRateConfig(rps=pipeline_cfg.rate_limit_rps),
        )
        oma_client = OmaClient(
            base_url="https://omabrowser.org/api",
            network=UniNetworkConfig(
                timeout_sec=pipeline_cfg.timeout_sec,
                max_retries=pipeline_cfg.retries,
            ),
            rate_limit=UniRateConfig(rps=pipeline_cfg.rate_limit_rps),
        )
        target_species = list(orth_cfg.get("target_species", []))
    return uni, hgnc, gtop, ens_client, oma_client, target_species


def main() -> None:
    """Run the unified pipeline on the provided input IDs."""

    args = parse_args()
    logging.basicConfig(level=args.log_level.upper())
    pipeline_cfg = load_pipeline_config(args.config)
    pipeline_cfg.list_format = args.list_format
    pipeline_cfg.species_priority = [args.species]
    pipeline_cfg.iuphar.affinity_parameter = args.affinity_parameter
    pipeline_cfg.iuphar.approved_only = (
        None
        if args.approved_only.lower() == "null"
        else args.approved_only.lower() == "true"
    )
    pipeline_cfg.iuphar.primary_target_only = args.primary_target_only.lower() == "true"
    pipeline_cfg.include_isoforms = args.with_isoforms

    # Load optional ChEMBL column configuration
    with open(args.config, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    chembl_cols = data.get("chembl", {}).get("columns")
    chembl_cfg: TargetConfig = TargetConfig(list_format=pipeline_cfg.list_format)
    if chembl_cols:
        chembl_cfg.columns = list(chembl_cols)

    (
        uni_client,
        hgnc_client,
        gtop_client,
        ens_client,
        oma_client,
        target_species,
    ) = build_clients(args.config, pipeline_cfg, with_orthologs=args.with_orthologs)

    df = pd.read_csv(args.input, sep=args.sep, encoding=args.encoding)
    if args.id_column not in df.columns:
        raise ValueError(f"Missing required column '{args.id_column}'")
    ids_series = df[args.id_column].astype(str).map(str.strip)
    ids_series = ids_series[ids_series != ""]
    ids: List[str] = list(dict.fromkeys(ids_series))

    out_df = run_pipeline(
        ids,
        pipeline_cfg,
        chembl_fetcher=fetch_targets,
        chembl_config=chembl_cfg,
        uniprot_client=uni_client,
        hgnc_client=hgnc_client,
        gtop_client=gtop_client,
        ensembl_client=ens_client,
        oma_client=oma_client,
        target_species=target_species,
    )
    out_df.to_csv(args.output, index=False, sep=args.sep, encoding=args.encoding)


if __name__ == "__main__":
    main()
