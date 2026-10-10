"""Render only validated, actually-measured search metrics into README."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- RESULTS START -->"
END = "<!-- RESULTS END -->"
METRICS: tuple[tuple[str, str, str], ...] = (
    ("Video Recall@5", "video_recall_at_5", ""),
    ("Moment Recall@5 (tIoU >= 0.5)", "moment_recall_at_5_tiou_0_5", ""),
    ("Search latency p95", "search_latency_p95_ms", " ms"),
    ("Index size per video-hour", "index_bytes_per_hour", " bytes/hour"),
)


def load_measurements(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Measurement record must be an object")
    if not isinstance(data.get("metrics"), dict):
        raise ValueError("Missing measured metrics")
    environment = data.get("environment")
    if not isinstance(environment, dict) or not environment.get("git_commit"):
        raise ValueError("Missing environment.git_commit provenance")
    measured_by = data.get("measured_by")
    if not isinstance(measured_by, str) or not measured_by.startswith(("bench/", "eval/")):
        raise ValueError("Missing bench/ or eval/ measurement script identifier")
    return data


def render(data: dict[str, Any] | None) -> str:
    lines = [START, "| Metric | Result |", "| --- | --- |"]
    measured = data["metrics"] if data is not None else {}
    for label, key, unit in METRICS:
        value = measured.get(key)
        if value is None:
            shown = "TBD (not yet measured)"
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Non-numeric measured metric: {key}")
        elif not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid measured metric: {key}")
        else:
            shown = f"{value:.3f}{unit}"
        lines.append(f"| {label} | {shown} |")
    return "\n".join([*lines, END])


def replace_table(readme: str, table: str) -> str:
    if readme.count(START) != 1 or readme.count(END) != 1:
        raise ValueError("README result markers must be unique")
    before, remainder = readme.split(START, 1)
    _, after = remainder.split(END, 1)
    return before + table + after


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if README table is stale")
    parser.add_argument("--result", type=Path, default=ROOT / "results/search_eval.json")
    args = parser.parse_args()
    readme = ROOT / "README.md"
    old = readme.read_text(encoding="utf-8")
    try:
        updated = replace_table(old, render(load_measurements(args.result)))
    except (ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.check:
        if updated != old:
            print("README measured results are stale; run scripts/render_results.py")
            return 1
        return 0
    readme.write_text(updated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
