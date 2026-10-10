"""Evaluate a user-supplied licensed corpus's frozen JSON annotations/predictions."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from eval.metrics import (
    Annotation,
    Prediction,
    hit_within,
    moment_recall,
    reciprocal_rank,
    video_recall,
)
from vsearch.environment import collect_environment


def compute(
    records: list[dict[str, object]],
) -> dict[str, float]:
    if not records:
        raise ValueError("No evaluation queries")
    by_metric: dict[str, list[float]] = {
        "video_recall_at_1": [],
        "video_recall_at_5": [],
        "video_recall_at_10": [],
        "moment_recall_at_5_tiou_0_3": [],
        "moment_recall_at_5_tiou_0_5": [],
        "video_mrr": [],
        "within_2s_at_5": [],
    }
    for record in records:
        annotations = [Annotation(**a) for a in record["annotations"]]  # type: ignore[arg-type]
        predictions = [Prediction(**p) for p in record["predictions"]]  # type: ignore[arg-type]
        by_metric["video_recall_at_1"].append(video_recall(annotations, predictions, k=1))
        by_metric["video_recall_at_5"].append(video_recall(annotations, predictions, k=5))
        by_metric["video_recall_at_10"].append(video_recall(annotations, predictions, k=10))
        by_metric["moment_recall_at_5_tiou_0_3"].append(
            moment_recall(annotations, predictions, k=5, threshold=0.3)
        )
        by_metric["moment_recall_at_5_tiou_0_5"].append(
            moment_recall(annotations, predictions, k=5, threshold=0.5)
        )
        by_metric["video_mrr"].append(reciprocal_rank(annotations, predictions))
        by_metric["within_2s_at_5"].append(hit_within(annotations, predictions))
    return {name: statistics.mean(values) for name, values in by_metric.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.queries.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Expected JSON list of query records")
    metrics = compute(payload)
    output = {
        "measured_by": "eval/evaluate.py",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "query_count": len(payload),
        "metrics": metrics,
        "environment": collect_environment(),
        "small_sample_warning": len(payload) < 100,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
