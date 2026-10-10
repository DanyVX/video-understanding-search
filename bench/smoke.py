"""Measure a deterministic synthetic text-index smoke test, not model quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from eval.metrics import Annotation, Prediction, moment_recall, video_recall
from vsearch.asr.transcript import Segment
from vsearch.environment import collect_environment
from vsearch.index.manifest import Manifest
from vsearch.index.transcripts import TranscriptIndex
from vsearch.retrieve.service import search_moments


def synthetic_id(index: int) -> str:
    return hashlib.sha256(f"synthetic-index-smoke-{index}".encode()).hexdigest()[:24]


def run(root: Path) -> dict[str, object]:
    """Exercise real SQLite/BM25/fusion code with fabricated non-media records.

    Do not cite these numbers as accuracy on actual video.
    """
    manifest = Manifest(root)
    transcripts = TranscriptIndex(root)
    for index in range(20):
        video_id = synthetic_id(index)
        manifest.upsert_video(
            video_id, f"synthetic-{index:02d}", None,
            hashlib.sha256(str(index).encode()).hexdigest(),
            8.0, 640, 360, False, "synthetic-test-v1",
        )
        manifest.set_status(video_id, "done")
        transcripts.replace(video_id, [
            Segment(2.0, 4.0, f"The artificial token marker_{index:02d} is announced.")
        ])
    timings_ms: list[float] = []
    video_scores: list[float] = []
    moment_scores: list[float] = []
    for index in range(20):
        target = synthetic_id(index)
        for template in (
            "marker_{:02d}", "find marker_{:02d}", "announced marker_{:02d}",
            "which clip marker_{:02d}", "artificial marker_{:02d}",
        ):
            query = template.format(index)
            start = time.perf_counter_ns()
            hits = search_moments(root, query, limit=10)
            timings_ms.append((time.perf_counter_ns() - start) / 1_000_000)
            ann = [Annotation(f"s-{index}", target, 2, 4)]
            predictions = [
                Prediction(hit.video_id, hit.start, hit.end, hit.score)
                for hit in hits
            ]
            video_scores.append(video_recall(ann, predictions, k=5))
            moment_scores.append(moment_recall(ann, predictions, k=5, threshold=0.5))
    unknown_hits = 0
    for i in range(10):
        hits = search_moments(root, f"notfound_unique_{i:02d}", limit=10)
        unknown_hits += int(bool(hits))
    disk_bytes = sum(
        item.stat().st_size for item in root.rglob("*") if item.is_file()
    )
    order = sorted(timings_ms)
    p95 = order[min(len(order) - 1, int(0.95 * (len(order) - 1)))]
    return {
        "measured_by": "bench/smoke.py",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "environment": collect_environment(),
        "corpus": "SYNTHETIC SQLITE TEXT FIXTURES - NOT ACTUAL VIDEO",
        "query_count": len(timings_ms) + 10,
        "video_count": 20,
        "real_video_accuracy_claim": False,
        "metrics": {
            "video_recall_at_5": statistics.mean(video_scores),
            "moment_recall_at_5_tiou_0_5": statistics.mean(moment_scores),
            "search_latency_p50_ms": statistics.median(timings_ms),
            "search_latency_p95_ms": p95,
            "synthetic_false_positive_fraction": unknown_hits / 10,
            "index_bytes": disk_bytes,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/synthetic-smoke.json"))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="vsearch-synthetic-") as name:
        report = run(Path(name))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("Synthetic text-index smoke complete (NOT a real video benchmark)")


if __name__ == "__main__":
    main()
