"""Resolve GtoPdb identifiers and dump related resources as CSV tables."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, cast

import pandas as pd
import yaml

if __package__ in {None, ""}:
    from _path_utils import ensure_project_root as _ensure_project_root

    _ensure_project_root()

from library.cli_common import (  # noqa: E402
    analyze_table_quality,
    ensure_output_dir,
    resolve_cli_sidecar_paths,
    serialise_dataframe,
    write_cli_metadata,
)
from library.gtop_client import GtoPClient, GtoPConfig, resolve_target  # noqa: E402
from library.gtop_normalize import (  # noqa: E402
    normalise_interactions,
    normalise_synonyms,
    normalise_targets,
)
from library.http_client import CacheConfig  # noqa: E402
from library.io import read_ids  # noqa: E402
from library.io_utils import CsvConfig, write_rows  # noqa: E402
from library.logging_utils import configure_logging  # noqa: E402

LOGGER = logging.getLogger("dump_gtop_target")
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "human"
DEFAULT_INPUT = "input.csv"
DEFAULT_ID_COLUMN = "uniprot_id"


def _default_output_dir(input_path: str) -> Path:
    """Return the default output directory derived from ``input_path``."""

    stem = Path(input_path).stem or "input"
    date_suffix = datetime.utcnow().strftime("%Y%m%d")
    return Path(f"output_{stem}_{date_suffix}")


def _load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file into a mutable dictionary."""

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Configuration in {path} is not a mapping")
    return dict(payload)


def _normalise_identifiers(values: List[str], column: str) -> List[str]:
    """Normalise raw identifier values for ``column``."""

    filtered = [value for value in values if value.lower() != "nan"]
    if column == "hgnc_id":
        return [value if value.startswith("HGNC:") else f"HGNC:{value}" for value in filtered]
    return filtered


def _read_identifiers(path: Path, column: str, cfg: CsvConfig) -> List[str]:
    """Read and normalise identifiers from ``column`` in ``path``."""

    normaliser = None if column in {"target_name", "gene_symbol"} else str.upper
    identifiers = list(read_ids(path, column, cfg, normalise=normaliser))
    return _normalise_identifiers(identifiers, column)


def _serialise_and_write(
    frame: pd.DataFrame, path: Path, cfg: CsvConfig, *, list_format: str
) -> None:
    """Serialise ``frame`` and persist it to ``path`` using :func:`write_rows`."""

    serialised = serialise_dataframe(frame, list_format, inplace=True)
    columns = list(serialised.columns)
    rows = serialised.to_dict(orient="records")
    write_rows(path, rows, columns, cfg)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the GtoP dump CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Input CSV file containing target identifiers",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write output tables. Defaults to output_<input>_<YYYYMMDD>",
    )
    parser.add_argument(
        "--id-column",
        choices=["uniprot_id", "target_name", "hgnc_id", "gene_symbol"],
        default=DEFAULT_ID_COLUMN,
        help="Identifier column present in the input CSV",
    )
    parser.add_argument(
        "--species",
        default="Human",
        help="Species filter applied to species-aware endpoints",
    )
    parser.add_argument(
        "--affinity-parameter",
        default="pKi",
        help="Affinity parameter for the interactions endpoint",
    )
    parser.add_argument(
        "--affinity-ge",
        type=float,
        default=None,
        help="Minimum affinity value for the interactions endpoint",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        help="Logging level (e.g. DEBUG, INFO)",
    )
    parser.add_argument(
        "--log-format",
        default=DEFAULT_LOG_FORMAT,
        choices=("human", "json"),
        help="Logging output format",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--sep",
        default=None,
        help="CSV delimiter for input/output (defaults to config or ',')",
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help="Text encoding for CSV files (defaults to config or 'utf-8-sig')",
    )
    parser.add_argument(
        "--meta-output",
        default=None,
        help="Optional path for the metadata YAML next to targets.csv",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for dumping GtoP resources."""

    args = parse_args(argv)

    configure_logging(args.log_level, log_format=args.log_format)

    config_path = Path(args.config).expanduser().resolve()
    cfg_dict = _load_config(config_path)
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

    output_section = cfg_dict.get("output")
    output_cfg = output_section if isinstance(output_section, Mapping) else {}
    config_sep = str(output_cfg.get("sep", ","))
    config_encoding = str(output_cfg.get("encoding", "utf-8-sig"))
    sep = args.sep if args.sep else config_sep
    encoding = args.encoding if args.encoding else config_encoding
    csv_cfg = CsvConfig(sep=sep, encoding=encoding, list_format="json")

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist")

    try:
        identifiers = _read_identifiers(input_path, args.id_column, csv_cfg)
    except KeyError as error:
        header = pd.read_csv(
            input_path,
            sep=csv_cfg.sep,
            encoding=csv_cfg.encoding,
            nrows=0,
        )
        available = ", ".join(header.columns)
        msg = (
            f"Column '{args.id_column}' not found in {input_path}."
            f" Available columns: {available or '<none>'}"
        )
        raise SystemExit(msg) from error

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (input_path.parent / _default_output_dir(args.input))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    targets: list[Dict[str, Any]] = []
    syn_rows: list[pd.DataFrame] = []
    int_rows: list[pd.DataFrame] = []
    for raw in identifiers:
        target = resolve_target(client, raw, args.id_column)
        if not target:
            continue
        targets.append(target)
        tid = cast(int, target.get("targetId"))
        synonyms = client.fetch_target_endpoint(tid, "synonyms")
        syn_rows.append(normalise_synonyms(tid, synonyms))
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

    targets_df = normalise_targets(targets)
    syn_df = (
        pd.concat(syn_rows, ignore_index=True)
        if syn_rows
        else pd.DataFrame(columns=["targetId", "synonym", "source"])
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

    targets_path = ensure_output_dir(output_dir / "targets.csv")
    _serialise_and_write(targets_df, targets_path, csv_cfg, list_format=csv_cfg.list_format)
    targets_meta, _, targets_quality = resolve_cli_sidecar_paths(
        targets_path,
        meta_output=args.meta_output,
    )
    analyze_table_quality(targets_df, table_name=str(targets_quality))

    syn_path = ensure_output_dir(output_dir / "targets_synonyms.csv")
    _serialise_and_write(syn_df, syn_path, csv_cfg, list_format=csv_cfg.list_format)
    _, _, syn_quality = resolve_cli_sidecar_paths(syn_path)
    analyze_table_quality(syn_df, table_name=str(syn_quality))

    int_path = ensure_output_dir(output_dir / "targets_interactions.csv")
    _serialise_and_write(int_df, int_path, csv_cfg, list_format=csv_cfg.list_format)
    _, _, int_quality = resolve_cli_sidecar_paths(int_path)
    analyze_table_quality(int_df, table_name=str(int_quality))
    write_cli_metadata(
        targets_path,
        row_count=int(targets_df.shape[0]),
        column_count=int(targets_df.shape[1]),
        namespace=args,
        meta_path=targets_meta,
    )

    print(targets_path)


if __name__ == "__main__":  # pragma: no cover
    main()
