"""Video and temporal retrieval metrics, exact denominator conventions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Annotation:
    query_id: str
    video_id: str
    start: float
    end: float
    answerable: bool = True

    def __post_init__(self) -> None:
        if not self.query_id or not self.video_id:
            raise ValueError("Missing query/video identifier")
        if not all(math.isfinite(x) for x in (self.start, self.end)):
            raise ValueError("Nonfinite time")
        if self.start < 0 or self.end < self.start:
            raise ValueError("Invalid time range")


@dataclass(frozen=True)
class Prediction:
    video_id: str
    start: float
    end: float
    score: float


def temporal_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Half-open intervals. Zero-duration events have IoU zero."""
    x0, x1 = a
    y0, y1 = b
    if not all(math.isfinite(v) for v in (x0, x1, y0, y1)) or x0 > x1 or y0 > y1:
        raise ValueError("Invalid temporal range")
    intersection = max(0.0, min(x1, y1) - max(x0, y0))
    union = max(x1, y1) - min(x0, y0)
    return intersection / union if union > 0 else 0.0


def video_recall(
    annotations: Sequence[Annotation],
    predictions: Sequence[Prediction],
    *,
    k: int = 5,
) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    targets = {a.video_id for a in annotations if a.answerable}
    if not targets:
        return 0.0
    return float(bool(targets.intersection(p.video_id for p in predictions[:k])))


def moment_recall(
    annotations: Sequence[Annotation],
    predictions: Sequence[Prediction],
    *,
    k: int = 5,
    threshold: float = 0.5,
) -> float:
    if k < 1 or not 0 <= threshold <= 1:
        raise ValueError("Invalid moment recall options")
    relevant = [a for a in annotations if a.answerable]
    if not relevant:
        return 0.0
    return float(
        any(
            a.video_id == p.video_id
            and temporal_iou((a.start, a.end), (p.start, p.end)) >= threshold
            for a in relevant
            for p in predictions[:k]
        )
    )


def reciprocal_rank(annotations: Sequence[Annotation], predictions: Sequence[Prediction]) -> float:
    relevant = {a.video_id for a in annotations if a.answerable}
    for rank, pred in enumerate(predictions, start=1):
        if pred.video_id in relevant:
            return 1.0 / rank
    return 0.0


def hit_within(
    annotations: Sequence[Annotation],
    predictions: Sequence[Prediction],
    *,
    tolerance: float = 2.0,
    k: int = 5,
) -> float:
    if tolerance < 0 or not math.isfinite(tolerance) or k < 1:
        raise ValueError("Invalid tolerance")
    return float(
        any(
            a.video_id == p.video_id
            and abs((a.start + a.end) / 2 - (p.start + p.end) / 2) <= tolerance
            for a in annotations
            if a.answerable
            for p in predictions[:k]
        )
    )


def bootstrap_interval(
    samples: Sequence[float], *, iterations: int = 1000, seed: int = 42
) -> tuple[float, float]:
    """Percentile CI over queries, seeded; descriptive only on small samples."""
    import random

    if not samples or iterations < 10:
        raise ValueError("Need samples and >=10 iterations")
    if any(not math.isfinite(v) for v in samples):
        raise ValueError("Nonfinite sample")
    generator = random.Random(seed)
    n = len(samples)
    estimates = sorted(
        sum(generator.choice(samples) for _ in range(n)) / n for _ in range(iterations)
    )
    return estimates[int(0.025 * (iterations - 1))], estimates[int(0.975 * (iterations - 1))]
