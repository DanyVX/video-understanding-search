"""Chunked, resumable frame sampling for media with bounded memory usage."""

from __future__ import annotations

import hashlib
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageStat

from vsearch.index.manifest import Manifest
from vsearch.ingest.probe import MediaError, probe, run_media_command, sha256_file

INGEST_VERSION = "m1-fps-0.5-v1"


@dataclass(frozen=True)
class IngestOptions:
    fps: float = 0.5
    chunk_seconds: int = 120
    keep_source: bool = False
    max_frames_per_chunk: int = 1000

    def validate(self) -> None:
        if not (0.05 <= self.fps <= 4):
            raise ValueError("fps must be between 0.05 and 4")
        if not (2 <= self.chunk_seconds <= 600):
            raise ValueError("chunk_seconds must be between 2 and 600")
        if not (1 <= self.max_frames_per_chunk <= 2400):
            raise ValueError("max_frames_per_chunk is outside allowed bounds")
        if self.chunk_seconds * self.fps > self.max_frames_per_chunk:
            raise ValueError("frame budget exceeded; reduce sampling or chunk size")


def _perceptual_hash(path: Path) -> tuple[str, float]:
    """Simple horizontal dHash and average luminance; avoids full-frame storage."""
    with Image.open(path) as image:
        small = image.convert("L").resize((9, 8))
        pixels = list(small.getdata())
        bits = 0
        for y in range(8):
            for x in range(8):
                bits = (bits << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
        luminance = float(ImageStat.Stat(small).mean[0])
    return f"{bits:016x}", luminance


def _distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def ingest(
    path: Path, manifest: Manifest, options: IngestOptions | None = None
) -> dict[str, object]:
    options = options or IngestOptions()
    options.validate()
    path = path.expanduser().resolve(strict=True)
    info = probe(path)
    hash_value = sha256_file(path)
    video_id = hash_value[:24]
    current = manifest.get_video(video_id)
    # Distinct filenames with identical byte content resolve to one logical item.
    if current and current["status"] == "done" and current["ingest_version"] == INGEST_VERSION:
        if not manifest.verify():
            return current
    manifest.upsert_video(
        video_id,
        path.name,
        str(path) if options.keep_source else None,
        hash_value,
        info.duration,
        info.width,
        info.height,
        info.has_audio,
        INGEST_VERSION,
    )
    manifest.set_status(video_id, "processing")
    target_dir = manifest.root / "thumbs" / video_id
    target_dir.mkdir(parents=True, exist_ok=True)
    total_chunks = math.ceil(info.duration / options.chunk_seconds)
    completed = manifest.chunks_done(video_id)
    try:
        for chunk in range(total_chunks):
            if chunk in completed:
                continue
            manifest.begin_chunk(video_id, chunk)
            start = chunk * options.chunk_seconds
            span = min(float(options.chunk_seconds), info.duration - start)
            stage_parent = manifest.root / "staging"
            stage_parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=stage_parent) as temp_name:
                temp = Path(temp_name)
                if not info.audio_only:
                    result = run_media_command(
                        [
                            "ffmpeg",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-nostdin",
                            "-ss",
                            str(start),
                            "-i",
                            str(path),
                            "-t",
                            str(span),
                            "-vf",
                            f"fps={options.fps},scale='min(480,iw)':-2",
                            "-q:v",
                            "5",
                            "-vsync",
                            "vfr",
                            str(temp / "frame_%06d.jpg"),
                        ],
                        timeout=max(60, span * 4),
                    )
                    if result.returncode != 0:
                        raise MediaError(f"Decoding failed in chunk {chunk}")
                rows: list[tuple[str, float, int, str, str]] = []
                prev_hash: str | None = None
                scene_id = chunk * 10000
                frames = sorted(temp.glob("frame_*.jpg"))
                if len(frames) > options.max_frames_per_chunk:
                    raise MediaError("Decoded more frames than the configured chunk budget")
                for index, frame in enumerate(frames):
                    try:
                        hash_hex, luminance = _perceptual_hash(frame)
                    except (OSError, ValueError):
                        continue  # Truncated frame, not a whole-ingest failure.
                    if luminance < 8 or luminance > 249:
                        continue  # Credits, black frames and blown-out fades.
                    if prev_hash is not None:
                        distance = _distance(prev_hash, hash_hex)
                        if distance <= 3:
                            continue
                        if distance >= 24:
                            scene_id += 1  # Sample-based scene heuristic, not full scene detect.
                    prev_hash = hash_hex
                    timestamp = min(info.duration, start + index / options.fps)
                    fid = hashlib.sha256(f"{video_id}:{timestamp:.3f}".encode()).hexdigest()[:24]
                    relative = Path("thumbs") / video_id / f"{fid}.jpg"
                    destination = manifest.root / relative
                    try:
                        shutil.copyfile(frame, destination.with_suffix(".tmp"))
                        destination.with_suffix(".tmp").replace(destination)
                    except OSError as exc:
                        raise MediaError("Failed to write thumbnail; check free disk") from exc
                    rows.append((fid, round(timestamp, 3), scene_id, hash_hex, relative.as_posix()))
                manifest.finish_chunk(video_id, chunk, rows)
        manifest.set_status(video_id, "done")
    except (OSError, MediaError) as exc:
        manifest.set_status(video_id, "failed", type(exc).__name__)
        raise
    result_video = manifest.get_video(video_id)
    if result_video is None:
        raise RuntimeError("Manifest unexpectedly lost video after ingest")
    return result_video


def delete_video(manifest: Manifest, video_id: str) -> bool:
    """No arbitrary path accepted; delete known, content-addressed derived files."""
    if not video_id or len(video_id) != 24 or any(c not in "0123456789abcdef" for c in video_id):
        return False
    # Remove rows first, then id-scoped artifacts. Verify/repair can detect stale files.
    existed = manifest.erase(video_id)
    if existed:
        shutil.rmtree(manifest.root / "thumbs" / video_id, ignore_errors=True)
    return existed
