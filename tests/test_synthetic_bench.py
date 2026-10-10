"""Synthetic smoke is a deterministic functionality test, not measured project accuracy."""

from pathlib import Path

from bench.smoke import run
from scripts.generate_demo import generate


def test_smoke_clearly_disclaims_real_video_results(tmp_path: Path) -> None:
    report = run(tmp_path)
    assert report["video_count"] == 20
    assert report["query_count"] == 110
    assert report["real_video_accuracy_claim"] is False
    assert "SYNTHETIC" in str(report["corpus"])
    assert report["metrics"]["video_recall_at_5"] >= 0


def test_generator_requires_binary(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.setattr("scripts.generate_demo.shutil.which", lambda name: None)  # type: ignore[attr-defined]
    try:
        generate(tmp_path, 1)
    except RuntimeError as exc:
        assert "ffmpeg" in str(exc)
    else:
        raise AssertionError("Expected missing-ffmpeg refusal")
