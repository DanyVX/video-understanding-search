"""Generate own synthetic, non-personal MP4 clips for a local ingestion demo.

Uses ffmpeg when installed, no downloads or third-party footage. Videos remain
untracked under data/synthetic. This does not establish real semantic accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

COLORS = ("red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta")


def generate(output: Path, count: int = 20) -> list[dict[str, object]]:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg must be installed separately")
    if not 1 <= count <= 20:
        raise ValueError("count must be between 1 and 20")
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index in range(count):
        color = COLORS[index % len(COLORS)]
        filename = f"synthetic-{index:02d}-{color}.mp4"
        path = output / filename
        subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c={color}:s=320x180:r=10:d=6",
            "-vf", f"drawbox=x={10 + index * 5}:y=60:w=40:h=40:color=white:t=fill",
            "-an", "-c:v", "mpeg4", "-q:v", "6", "-y", str(path),
        ], check=True, timeout=40, stdout=subprocess.DEVNULL,
           stderr=subprocess.PIPE)
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        records.append({
            "filename": filename, "sha256": h.hexdigest(), "duration_seconds": 6,
            "resolution": "320x180", "provenance": "generated locally by script",
            "license": "self-created synthetic demonstration; no humans depicted",
        })
    (output / "manifest.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    print(f"Generated {len(generate(args.output, args.count))} synthetic clips")


if __name__ == "__main__":
    main()
