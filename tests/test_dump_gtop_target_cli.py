from __future__ import annotations

import csv
from pathlib import Path
import sys

from collections.abc import Callable, Iterator

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import dump_gtop_target  # noqa: E402


class DummyClient:
    """Minimal stub emulating ``GtoPClient`` interactions."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.synonym_calls: list[int] = []
        self.interaction_calls: list[int] = []

    def fetch_target_endpoint(
        self, target_id: int, endpoint: str, params: dict[str, object] | None = None
    ) -> list[dict[str, object]]:
        if endpoint == "synonyms":
            self.synonym_calls.append(target_id)
            return [{"synonym": "Example", "synonymType": "Preferred"}]
        if endpoint == "interactions":
            self.interaction_calls.append(target_id)
            return [
                {
                    "ligandId": 42,
                    "type": "Inhibition",
                    "action": "antagonist",
                    "affinity": 8.7,
                    "affinityParameter": params.get("affinityType") if params else None,
                    "species": "Human",
                    "ligandType": "Small molecule",
                    "approved": True,
                    "primaryTarget": True,
                }
            ]
        msg = f"Unexpected endpoint: {endpoint}"
        raise AssertionError(msg)


def test_dump_gtop_target_cli_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the GtoP dump CLI writes CSV tables and metadata."""

    input_csv = tmp_path / "ids.csv"
    input_csv.write_text("uniprot_id\np12345\n", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "gtop:",
                "  base_url: https://example.org",  # mocked HTTP interactions
                "output:",
                "  encoding: utf-8",
                "  sep: ','",
            ]
        ),
        encoding="utf-8",
    )

    dummy_client = DummyClient()
    monkeypatch.setattr(
        dump_gtop_target, "GtoPClient", lambda *_args, **_kwargs: dummy_client
    )
    monkeypatch.setattr(
        dump_gtop_target,
        "resolve_target",
        lambda _client, identifier, _column: {
            "targetId": 101,
            "name": f"Target {identifier.upper()}",
            "targetType": "Protein",
            "family": "GPCR",
            "species": "Human",
            "description": "Mock target",
        },
    )

    output_dir = tmp_path / "gtop"
    meta_file = output_dir / "targets_overview.meta.yaml"

    dump_gtop_target.main(
        [
            "--input",
            str(input_csv),
            "--output-dir",
            str(output_dir),
            "--config",
            str(config_path),
            "--id-column",
            "uniprot_id",
            "--encoding",
            "utf-8",
            "--sep",
            ",",
            "--meta-output",
            str(meta_file),
        ]
    )

    targets_csv = output_dir / "targets.csv"
    assert targets_csv.exists()

    with targets_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert rows == [
        {
            "targetId": "101",
            "name": "Target P12345",
            "targetType": "Protein",
            "family": "GPCR",
            "species": "Human",
            "description": "Mock target",
        }
    ]

    assert dummy_client.synonym_calls == [101]
    assert dummy_client.interaction_calls == [101]

    assert meta_file.exists()
    meta = yaml.safe_load(meta_file.read_text(encoding="utf-8"))
    assert meta["rows"] == 1
    assert meta["columns"] == 6
    assert meta["output"] == str(targets_csv)


def test_dump_gtop_target_cli_many_identifiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI streams and normalises many identifiers without exhaustion."""

    total = 250

    def fake_read_ids(
        _path: Path,
        column: str,
        _cfg: dump_gtop_target.CsvConfig,
        *,
        limit: int | None = None,
        normalise: Callable[[str], str] | None = str.upper,
    ) -> Iterator[str]:
        del limit
        assert column == "hgnc_id"
        # Yield numeric identifiers lazily while mimicking the normalisation hook.
        for value in map(str, range(total)):
            yield normalise(value) if normalise is not None else value

    config_path = tmp_path / "config.yaml"
    config_path.write_text("gtop: {}\n", encoding="utf-8")

    input_csv = tmp_path / "ids.csv"
    input_csv.write_text("hgnc_id\n", encoding="utf-8")

    dummy_client = DummyClient()
    monkeypatch.setattr(
        dump_gtop_target, "GtoPClient", lambda *_args, **_kwargs: dummy_client
    )
    monkeypatch.setattr(dump_gtop_target, "read_ids", fake_read_ids)

    received: list[int] = []

    def fake_resolve_target(
        _client: dump_gtop_target.GtoPClient, identifier: str, column: str
    ) -> dict[str, object]:
        assert column == "hgnc_id"
        assert identifier.startswith("HGNC:")
        idx = int(identifier.split(":", 1)[1])
        received.append(idx)
        return {
            "targetId": idx,
            "name": f"Target {idx}",
            "targetType": "Protein",
            "family": "GPCR",
            "species": "Human",
            "description": "Mock target",
        }

    monkeypatch.setattr(dump_gtop_target, "resolve_target", fake_resolve_target)

    output_dir = tmp_path / "bulk"

    dump_gtop_target.main(
        [
            "--input",
            str(input_csv),
            "--output-dir",
            str(output_dir),
            "--config",
            str(config_path),
            "--id-column",
            "hgnc_id",
        ]
    )

    assert received == list(range(total))
    assert dummy_client.synonym_calls == list(range(total))
    assert dummy_client.interaction_calls == list(range(total))

    targets_csv = output_dir / "targets.csv"
    with targets_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert len(rows) == total
