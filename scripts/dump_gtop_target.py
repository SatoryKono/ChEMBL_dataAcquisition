"""Download comprehensive GtoPdb target information for a list of identifiers."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, cast

import pandas as pd
import yaml

if __package__ in {None, ""}:
    from _path_utils import ensure_project_root as _ensure_project_root

    _ensure_project_root()

from library.data_profiling import analyze_table_quality  # noqa: E402
from library.logging_utils import configure_logging  # noqa: E402
from library.gtop_client import GtoPClient, GtoPConfig, resolve_target  # noqa: E402
from library.http_client import CacheConfig  # noqa: E402
from library.gtop_normalize import (  # noqa: E402
    normalise_interactions,
    normalise_synonyms,
    normalise_targets,
)

LOGGER = logging.getLogger("dump_gtop_target")
DEFAULT_LOG_FORMAT = "human"


def _load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Configuration in {path} is not a mapping")
    return dict(data)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        An `argparse.Namespace` object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write output tables"
    )
    parser.add_argument(
        "--id-column",
        required=True,
        choices=["uniprot_id", "target_name", "hgnc_id", "gene_symbol"],
        help="Identifier column present in the input CSV",
    )
    parser.add_argument(
        "--species",
        default="Human",
        help="Species filter passed to species-aware endpoints",
    )
    parser.add_argument(
        "--affinity-parameter",
        default="pKi",
        help="Affinity parameter for interactions endpoint",
    )
    parser.add_argument(
        "--affinity-ge",
        type=float,
        default=None,
        help="Minimum affinity value for interactions endpoint",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (e.g. DEBUG)"
    )
    parser.add_argument(
        "--log-format",
        default=DEFAULT_LOG_FORMAT,
        choices=("human", "json"),
        help="Logging output format (human or json)",
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to YAML configuration"
    )
    return parser.parse_args()


def read_ids(path: Path, column: str) -> list[str]:
    """Reads and normalizes a list of identifiers from a CSV file.

    Args:
        path: The path to the CSV file.
        column: The name of the column containing the identifiers.

    Returns:
        A list of normalized identifiers.
    """
    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in input")
    series = df[column].astype(str).str.strip()
    if column == "uniprot_id":
        series = series.str.upper()
    if column == "hgnc_id":
        series = series.str.upper().apply(
            lambda x: x if x.startswith("HGNC:") else f"HGNC:{x}"
        )
    ids = list(dict.fromkeys([x for x in series if x and x != "nan"]))
    return ids


def main() -> None:
    """The main entry point for the script."""
    args = parse_args()
    configure_logging(args.log_level, log_format=args.log_format)
    cfg_dict = _load_config(Path(args.config))
    gtop_section = cfg_dict.get("gtop", {})
    if not isinstance(gtop_section, Mapping):
        raise ValueError("'gtop' configuration must be a mapping")
    gcfg: Dict[str, Any] = dict(gtop_section)
    http_cache_cfg = cfg_dict.get("http_cache")
    http_cache_mapping = http_cache_cfg if isinstance(http_cache_cfg, Mapping) else None
    global_cache = CacheConfig.from_dict(http_cache_mapping)
    cache_cfg = gcfg.get("cache")
    cache_mapping = cache_cfg if isinstance(cache_cfg, Mapping) else None
    client = GtoPClient(
        GtoPConfig(
            base_url=gcfg.get(
                "base_url", "https://www.guidetopharmacology.org/services"
            ),
            timeout_sec=cfg_dict.get("network", {}).get("timeout_sec", 30),
            max_retries=cfg_dict.get("network", {}).get("max_retries", 3),
            rps=cfg_dict.get("rate_limit", {}).get("rps", 2),
            cache=CacheConfig.from_dict(cache_mapping) or global_cache,
        )
    )

    ids = read_ids(Path(args.input), args.id_column)
    targets: list[Dict[str, Any]] = []
    syn_rows: list[pd.DataFrame] = []
    int_rows: list[pd.DataFrame] = []
    for raw in ids:
        target = resolve_target(client, raw, args.id_column)
        if not target:
            continue
        targets.append(target)
        tid = cast(int, target.get("targetId"))
        syn = client.fetch_target_endpoint(tid, "synonyms")
        syn_rows.append(normalise_synonyms(tid, syn))
        interactions = client.fetch_target_endpoint(
            tid,
            "interactions",
            params={
                "affinityType": args.affinity_parameter,
                "affinity": args.affinity_ge,
                "species": args.species,
            },
        )
        int_rows.append(normalise_interactions(tid, interactions))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets_df = normalise_targets(targets)
    targets_df.to_csv(
        out_dir / "targets.csv",
        index=False,
        encoding=cfg_dict.get("output", {}).get("encoding", "utf-8-sig"),
        sep=cfg_dict.get("output", {}).get("sep", ","),
        quoting=csv.QUOTE_MINIMAL,
    )
    analyze_table_quality(
        targets_df, table_name=str((out_dir / "targets").with_suffix(""))
    )
    syn_df = (
        pd.concat(syn_rows, ignore_index=True)
        if syn_rows
        else pd.DataFrame(columns=["targetId", "synonym", "source"])
    )
    syn_df.to_csv(
        out_dir / "targets_synonyms.csv",
        index=False,
        encoding=cfg_dict.get("output", {}).get("encoding", "utf-8-sig"),
        sep=cfg_dict.get("output", {}).get("sep", ","),
        quoting=csv.QUOTE_MINIMAL,
    )
    analyze_table_quality(
        syn_df, table_name=str((out_dir / "targets_synonyms").with_suffix(""))
    )
    int_df = (
        pd.concat(int_rows, ignore_index=True)
        if int_rows
        else pd.DataFrame(
            columns=[
                "targetId",
                "ligandId",
                "type",
                "action",
                "affinity",
                "affinityParameter",
                "species",
                "ligandType",
                "approved",
                "primaryTarget",
            ]
        )
    )
    int_df.to_csv(
        out_dir / "targets_interactions.csv",
        index=False,
        encoding=cfg_dict.get("output", {}).get("encoding", "utf-8-sig"),
        sep=cfg_dict.get("output", {}).get("sep", ","),
        quoting=csv.QUOTE_MINIMAL,
    )
    analyze_table_quality(
        int_df, table_name=str((out_dir / "targets_interactions").with_suffix(""))
    )


if __name__ == "__main__":
    main()
