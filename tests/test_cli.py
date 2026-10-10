"""Only small, CPU-only checks run in CI."""

import json

import pytest
from typer.testing import CliRunner

from vsearch.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == "0.1.0a0"


def test_doctor_machine_readable() -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["internet"] == "not-checked"
    assert report["python"]
    assert "gpu" in report
    assert "ffprobe" in report


def test_doctor_human_readable() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "library_versions:" in result.output


def test_bad_env_does_not_leak_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VSEARCH_MAX_UPLOAD_BYTES", "-123")
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 2
    assert "Invalid VSEARCH_ configuration" in result.output
    assert "-123" not in result.output
