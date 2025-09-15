"""Command line entry point for fetching PubMed metadata.

This script reads a CSV file containing a column of PMIDs, downloads metadata
using :mod:`library.pubmed_client` and writes a deterministic CSV with the
results.  Only a minimal subset of fields is retrieved to keep the example
lightweight.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd

from library.http_client import HttpClient
from library.pubmed_client import classify_publication, fetch_pubmed_records
from library.semantic_scholar_client import fetch_semantic_scholar_records
from library.openalex_client import fetch_openalex_records
from library.crossref_client import fetch_crossref_records


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch PubMed metadata")
    parser.add_argument("--input", default="input.csv", help="Input CSV path")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path; defaults to derived name",
    )
    parser.add_argument(
        "--column", default="PMID", help="Name of the column with PubMed IDs"
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.34,
        help="Delay between PubMed requests in seconds (approximate)",
    )
    parser.add_argument(
        "--openalex-rps",
        type=float,
        default=1.0,
        help="Maximum requests per second to OpenAlex",
    )
    parser.add_argument(
        "--crossref-rps",
        type=float,
        default=1.0,
        help="Maximum requests per second to Crossref",
    )
    parser.add_argument("--sep", default=",", help="CSV separator")
    parser.add_argument("--encoding", default="utf-8", help="CSV encoding")
    return parser


def _determine_output_path(input_path: Path, output: str | None) -> Path:
    if output:
        return Path(output)
    stem = input_path.stem
    date = datetime.utcnow().strftime("%Y%m%d")
    return Path(f"output_{stem}_{date}.csv")


def run(
    input_path: Path,
    output_path: Path,
    column: str,
    batch_size: int,
    sleep: float,
    openalex_rps: float,
    crossref_rps: float,
    sep: str,
    encoding: str,
) -> None:
    """Orchestrate downloads and write the final CSV."""

    df = pd.read_csv(input_path, sep=sep, encoding=encoding)
    if column not in df.columns:
        raise SystemExit(f"Column '{column}' not found in input")
    pmids: Sequence[str] = df[column].astype(str).tolist()

    pubmed_client = HttpClient(
        timeout=10.0, max_retries=3, rps=1.0 / sleep if sleep > 0 else 0
    )
    scholar_records = fetch_semantic_scholar_records(pmids, client=pubmed_client)
    pubmed_records = fetch_pubmed_records(
        pmids, client=pubmed_client, batch_size=batch_size
    )
    openalex_client = HttpClient(timeout=10.0, max_retries=3, rps=openalex_rps)
    openalex_records = fetch_openalex_records(pmids, client=openalex_client)
    # Gather DOIs from all sources for Crossref
    dois = (
        [r.doi for r in pubmed_records if r.doi]
        + [r.doi for r in scholar_records if r.doi]
        + [r.doi for r in openalex_records if r.doi]
    )
    unique_dois = sorted(set(dois))
    crossref_client = HttpClient(timeout=10.0, max_retries=3, rps=crossref_rps)
    crossref_records = fetch_crossref_records(unique_dois, client=crossref_client)
    scholar_map = {r.pmid: r for r in scholar_records}
    openalex_map = {r.pmid: r for r in openalex_records}
    crossref_map = {r.doi: r for r in crossref_records}

    rows = []
    for rec in pubmed_records:
        data = rec.to_dict()
        pub_types = list(rec.publication_types)
        scholar = scholar_map.get(rec.pmid)
        if scholar:
            data.update(scholar.to_dict())
            pub_types.extend(scholar.publication_types)
        openalex = openalex_map.get(rec.pmid)
        if openalex:
            data.update(openalex.to_dict())
            pub_types.extend(openalex.publication_types)
        doi = (
            data.get("PubMed.DOI")
            or data.get("scholar.DOI")
            or data.get("OpenAlex.DOI")
        )
        if isinstance(doi, str):
            cross = crossref_map.get(doi)
            if cross:
                data.update(cross.to_dict())
        data["publication_class"] = classify_publication(pub_types)
        rows.append(data)
    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values("PubMed.PMID").reset_index(drop=True)
    out_df.to_csv(output_path, sep=sep, index=False, encoding=encoding)
    LOGGER.info("Wrote %d rows to %s", len(out_df), output_path)


LOGGER = logging.getLogger("pubmed_main")


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    input_path = Path(args.input)
    output_path = _determine_output_path(input_path, args.output)
    run(
        input_path=input_path,
        output_path=output_path,
        column=args.column,
        batch_size=args.batch_size,
        sleep=args.sleep,
        openalex_rps=args.openalex_rps,
        crossref_rps=args.crossref_rps,
        sep=args.sep,
        encoding=args.encoding,
    )


if __name__ == "__main__":
    main()
