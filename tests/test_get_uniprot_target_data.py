from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "get_uniprot_target_data.py"


def test_missing_column_exits_cleanly(tmp_path: Path) -> None:
    """The UniProt target CLI should exit gracefully when the column is absent."""

    input_csv = tmp_path / "input.csv"
    input_csv.write_text("identifier\nP12345\n", encoding="utf-8")
    output_csv = tmp_path / "results.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(input_csv),
            "--output",
            str(output_csv),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "does not contain the required" in result.stderr
    assert "Traceback" not in result.stderr

    meta_path = output_csv.with_name(f"{output_csv.name}.meta.yaml")
    assert meta_path.exists()
    metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "error"
    assert "does not contain the required" in metadata["error"]
    assert metadata["sha256"] is None


def test_missing_input_file_exits_with_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """The CLI should report when the input file cannot be located."""

    import scripts.get_uniprot_target_data as cli

    real_path = cli.Path

    class MissingPathStub:
        """Stand-in path reporting that the referenced file is absent."""

        def __init__(self, raw: str):
            self._raw = raw

        def exists(self) -> bool:
            return False

        def is_file(self) -> bool:
            return False

        def __str__(self) -> str:
            return self._raw

        def __fspath__(self) -> str:
            return self._raw

    def fake_path(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if args and args[0] == "missing.csv" and len(args) == 1:
            return MissingPathStub("missing.csv")
        return real_path(*args, **kwargs)

    monkeypatch.setattr(cli, "Path", fake_path)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--input", "missing.csv", "--output", str(tmp_path / "output.csv")])

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Input file missing.csv does not exist" in captured.err
