"""Diagnostics stay bounded and do not leak git identity or paths."""

import subprocess

import pytest

from vsearch import environment


def test_run_missing_executable() -> None:
    assert environment._output(["this-binary-does-not-exist-6b67e"]) is None


def test_run_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="tool", timeout=1)

    monkeypatch.setattr(subprocess, "run", timeout)
    assert environment._output(["tool"]) is None


def test_report_offline_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_network() -> str:
        raise AssertionError("Unexpected network access")

    monkeypatch.setattr(environment, "_network_status", no_network)
    report = environment.collect_environment()
    assert report["internet"] == "not-checked"
    assert isinstance(report["free_disk_mb"], int)
    assert report["git_commit"]
    assert "library_versions" in report
    assert "email" not in report
