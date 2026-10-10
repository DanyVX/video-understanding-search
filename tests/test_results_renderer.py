"""Unmeasured values can never silently become benchmark claims."""

from pathlib import Path

import pytest

from scripts.render_results import load_measurements, render, replace_table


def test_no_benchmarks_means_only_tbd() -> None:
    output = render(None)
    assert output.count("TBD (not yet measured)") == 4


def test_invalid_measurement_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid measured metric"):
        render({"metrics": {"search_latency_p95_ms": -1}})


def test_provenance_is_required(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    result.write_text('{"metrics":{"video_recall_at_5":1.0}}', encoding="utf-8")
    with pytest.raises(ValueError, match="environment.git_commit"):
        load_measurements(result)


def test_marker_integrity() -> None:
    with pytest.raises(ValueError, match="unique"):
        replace_table("README without markers", render(None))
