"""CLI entry point for the unified target data pipeline."""

from __future__ import annotations

import argparse
import logging
from typing import List

import pandas as pd
import yaml  # type: ignore[import]

from chembl_targets import fetch_targets
from gtop_client import GtoPClient, GtoPConfig
from hgnc_client import HGNCClient, load_config as load_hgnc_config
from uniprot_client import (
    NetworkConfig as UniNetworkConfig,
    RateLimitConfig as UniRateConfig,
    UniProtClient,
)

from library.pipeline_targets import PipelineConfig, load_pipeline_config, run_pipeline


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def build_clients(
    cfg_path: str, pipeline_cfg: PipelineConfig
) -> tuple[UniProtClient, HGNCClient, GtoPClient]:
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

    hcfg = load_hgnc_config(cfg_path)
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
    return uni, hgnc, gtop


def main() -> None:
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

    uni_client, hgnc_client, gtop_client = build_clients(args.config, pipeline_cfg)

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
        uniprot_client=uni_client,
        hgnc_client=hgnc_client,
        gtop_client=gtop_client,
    )
    out_df.to_csv(args.output, index=False, sep=args.sep, encoding=args.encoding)


if __name__ == "__main__":
    main()
