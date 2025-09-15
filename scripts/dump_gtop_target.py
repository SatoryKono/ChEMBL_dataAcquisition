"""Download comprehensive GtoPdb target information for a list of identifiers."""

from __future__ import annotations

import argparse
import csv
import logging

import sys


from pathlib import Path
from typing import List, cast

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from library.gtop_client import GtoPClient, GtoPConfig, resolve_target  # noqa: E402
from library.gtop_normalize import (  # noqa: E402
    normalise_interactions,
    normalise_synonyms,
    normalise_targets,
)

LOGGER = logging.getLogger("dump_gtop_target")


def _load_config(path: Path) -> dict:
    """Load a YAML configuration file."""
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
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
        "--config", default="config.yaml", help="Path to YAML configuration"
    )
    return parser.parse_args()


def read_ids(path: Path, column: str) -> List[str]:
    """Read and normalize a list of identifiers from a CSV file."""
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
    """Main entry point for the script."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    cfg_dict = _load_config(Path(args.config))
    gcfg = cfg_dict.get("gtop", {})
    client = GtoPClient(
        GtoPConfig(
            base_url=gcfg.get(
                "base_url", "https://www.guidetopharmacology.org/services"
            ),
            timeout_sec=cfg_dict.get("network", {}).get("timeout_sec", 30),
            max_retries=cfg_dict.get("network", {}).get("max_retries", 3),
            rps=cfg_dict.get("rate_limit", {}).get("rps", 2),
        )
    )

    ids = read_ids(Path(args.input), args.id_column)
    targets = []
    syn_rows = []
    int_rows = []
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


if __name__ == "__main__":
    main()
