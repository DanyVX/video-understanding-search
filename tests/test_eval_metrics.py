"""Known mathematical answers. Synthetic fixtures are NOT real corpus results."""

import math

import pytest

from eval.metrics import (
    Annotation, Prediction, bootstrap_interval, hit_within, moment_recall,
    reciprocal_rank, temporal_iou, video_recall,
)
from eval.evaluate import compute


def test_iou_exact() -> None:
    assert temporal_iou((0, 10), (5, 15)) == pytest.approx(1 / 3)
    assert temporal_iou((1, 2), (3, 4)) == 0
    assert temporal_iou((1, 1), (1, 1)) == 0


def test_video_and_moment_separate() -> None:
    annotations = [Annotation("q1", "a", 1, 3)]
    wrong_moment = [Prediction("a", 40, 50, 10)]
    assert video_recall(annotations, wrong_moment, k=1) == 1
    assert moment_recall(annotations, wrong_moment, k=1) == 0
    assert reciprocal_rank(annotations, [Prediction("b", 1, 2, 3),
                                         Prediction("a", 1, 2, 2)]) == 0.5
    assert hit_within(annotations, wrong_moment) == 0


def test_bootstrap_reproducible() -> None:
    assert bootstrap_interval([0, 1, 1, 0], iterations=100, seed=5) == (
        bootstrap_interval([0, 1, 1, 0], iterations=100, seed=5)
    )


def test_validation() -> None:
    with pytest.raises(ValueError):
        temporal_iou((math.nan, 4), (0, 4))
    with pytest.raises(ValueError):
        Annotation("q", "v", 5, 3)
    with pytest.raises(ValueError):
        bootstrap_interval([])


def test_evaluator_works_without_video() -> None:
    rows = [{
        "annotations": [{"query_id": "q", "video_id": "v",
                         "start": 1, "end": 3}],
        "predictions": [{"video_id": "v", "start": 1, "end": 3, "score": 1}],
    }]
    scores = compute(rows)
    assert scores["video_recall_at_5"] == 1
    assert scores["moment_recall_at_5_tiou_0_5"] == 1
