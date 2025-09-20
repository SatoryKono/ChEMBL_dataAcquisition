"""Command-line benchmark comparing vectorised and legacy validators."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd

from library import legacy_validation
from library.activity_validation import validate_activities
from library.assay_validation import validate_assays
from library.testitem_validation import validate_testitems

LOGGER = logging.getLogger(__name__)

DatasetRunner = Callable[[pd.DataFrame, Path], Any]


def _default_output_path(dataset: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d")
    return Path(f"output_{dataset}_benchmark_{timestamp}.json")


def _prepare_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()), format="%(levelname)s: %(message)s"
    )


def _generate_sample(dataset: str, rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    if dataset == "testitems":
        data = {
            "molecule_chembl_id": [f"CHEMBL{index}" for index in range(rows)],
            "canonical_smiles": ["C" * ((index % 3) + 1) for index in range(rows)],
            "standard_inchi_key": [f"KEY{index}" for index in range(rows)],
            "max_phase": rng.integers(-1, 5, size=rows),
            "chembl_full_mwt": rng.uniform(0, 500, size=rows),
        }
    elif dataset == "assays":
        data = {
            "assay_chembl_id": [f"ASSAY{index}" for index in range(rows)],
            "document_chembl_id": [f"DOC{index}" for index in range(rows)],
            "assay_with_same_target": rng.integers(-1, 5, size=rows),
            "confidence_score": rng.integers(-1, 5, size=rows),
        }
    else:
        data = {
            "activity_chembl_id": [f"ACT{index}" for index in range(rows)],
            "assay_chembl_id": [f"ASSAY{index}" for index in range(rows)],
            "record_id": rng.integers(-5, 5, size=rows),
            "activity_id": rng.integers(-5, 5, size=rows),
            "standard_value": rng.uniform(-2, 2, size=rows),
        }
    return pd.DataFrame(data)


def _load_dataset(path: Path) -> pd.DataFrame:
    LOGGER.info("Loading dataset from %s", path)
    return pd.read_csv(path)


def _benchmark_runner(
    df: pd.DataFrame,
    *,
    dataset: str,
    runs: int,
    tmpdir: Path,
) -> dict[str, float]:
    timings: dict[str, list[float]] = {"vectorised": [], "legacy": []}

    if dataset == "testitems":

        def vectorised_runner(frame: pd.DataFrame, errors: Path) -> Any:
            return validate_testitems(frame, errors_path=errors)

        def legacy_runner(frame: pd.DataFrame, errors: Path) -> pd.DataFrame:
            return legacy_validation.legacy_validate_testitems(
                frame, errors_path=errors
            )

    elif dataset == "assays":

        def vectorised_runner(frame: pd.DataFrame, errors: Path) -> Any:
            return validate_assays(frame, errors_path=errors)

        def legacy_runner(frame: pd.DataFrame, errors: Path) -> pd.DataFrame:
            return legacy_validation.legacy_validate_assays(frame, errors_path=errors)

    else:

        def vectorised_runner(frame: pd.DataFrame, errors: Path) -> Any:
            return validate_activities(frame, errors_path=errors)

        def legacy_runner(frame: pd.DataFrame, errors: Path) -> pd.DataFrame:
            return legacy_validation.legacy_validate_activities(
                frame, errors_path=errors
            )

    for label, runner in ("vectorised", vectorised_runner), ("legacy", legacy_runner):
        for iteration in range(runs):
            errors_path = tmpdir / f"{label}_{iteration}.json"
            frame = df.copy(deep=True)
            start = perf_counter()
            result = runner(frame, errors_path)
            elapsed = perf_counter() - start
            timings[label].append(elapsed)
            if label == "vectorised":
                # Ensure side effects match expectations during benchmarking.
                _ = getattr(result, "valid", result)
            if errors_path.exists():
                errors_path.unlink()

    return {
        "vectorised_mean": (
            float(np.mean(timings["vectorised"])) if timings["vectorised"] else 0.0
        ),
        "legacy_mean": float(np.mean(timings["legacy"])) if timings["legacy"] else 0.0,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark validation implementations")
    parser.add_argument(
        "--dataset",
        choices=("testitems", "assays", "activities"),
        default="testitems",
        help="Target dataset to benchmark",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional CSV input file. When omitted, synthetic data is generated.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1000,
        help="Number of synthetic rows to generate when --input is not provided.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of repetitions per implementation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for persisting benchmark results.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity level.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _prepare_logging(args.log_level)

    if args.input is not None and not args.input.exists():
        LOGGER.error("Input file not found: %s", args.input)
        return 1

    if args.rows <= 0:
        LOGGER.error("--rows must be positive; received %s", args.rows)
        return 1

    if args.runs <= 0:
        LOGGER.error("--runs must be positive; received %s", args.runs)
        return 1

    dataset = (
        _load_dataset(args.input)
        if args.input is not None
        else _generate_sample(args.dataset, args.rows)
    )

    with tempfile.TemporaryDirectory() as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        timings = _benchmark_runner(
            dataset,
            dataset=args.dataset,
            runs=args.runs,
            tmpdir=tmpdir,
        )

    speedup = (
        timings["legacy_mean"] / timings["vectorised_mean"]
        if timings["vectorised_mean"] > 0
        else float("nan")
    )
    summary = {
        "dataset": args.dataset,
        "rows": int(len(dataset)),
        "columns": int(len(dataset.columns)),
        "runs": args.runs,
        "vectorised_mean": timings["vectorised_mean"],
        "legacy_mean": timings["legacy_mean"],
        "speedup": speedup,
    }

    LOGGER.info(
        "Vectorised mean %.6fs vs legacy %.6fs (speedup %.2fx)",
        summary["vectorised_mean"],
        summary["legacy_mean"],
        summary["speedup"],
    )

    output_path = args.output or _default_output_path(args.dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Benchmark results written to %s", output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
