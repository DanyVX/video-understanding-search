"""ffprobe metadata validation, safe subprocess calls and SHA-256 identity."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class MediaError(ValueError):
    """Media is unsupported, invalid or not safely decodable."""


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    width: int
    height: int
    has_audio: bool
    rotation: int = 0
    video_codec: str = ""
    audio_only: bool = False


def run_media_command(
    args: list[str], *, timeout: float = 120
) -> subprocess.CompletedProcess[bytes]:
    """Never invoke a shell: Unicode and adversarial filenames are literal arguments."""
    try:
        return subprocess.run(args, check=False, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise MediaError("Media tool timed out") from exc
    except FileNotFoundError as exc:
        raise MediaError(f"Required media tool not found: {args[0]}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path, *, executable: str = "ffprobe") -> MediaInfo:
    if not path.is_file():
        raise MediaError("Media file does not exist")
    if path.stat().st_size == 0:
        raise MediaError("Empty media file")
    if shutil.which(executable) is None:
        raise MediaError("ffprobe is unavailable; install it locally before ingest")
    result = run_media_command(
        [executable, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        timeout=30,
    )
    if result.returncode:
        raise MediaError("ffprobe rejected media: unsupported, DRM-protected or corrupt")
    try:
        metadata: dict[str, Any] = json.loads(result.stdout)
        streams: list[dict[str, Any]] = metadata.get("streams", [])
        videos = [stream for stream in streams if stream.get("codec_type") == "video"]
        audio = any(s.get("codec_type") == "audio" for s in streams)
        if not videos and not audio:
            raise MediaError("No supported audio or video streams")
        duration = float(metadata.get("format", {}).get("duration") or 0)
        if duration <= 0:
            duration = max((float(s.get("duration", 0)) for s in streams), default=0)
        if not math.isfinite(duration) or duration <= 0:
            raise MediaError("Unknown or invalid media duration")
        video = videos[0] if videos else {}
        width, height = int(video.get("width") or 0), int(video.get("height") or 0)
        if videos and (width < 1 or height < 1):
            raise MediaError("Invalid video geometry")
        side_data = video.get("side_data_list", [])
        rotations = [d.get("rotation") for d in side_data if "rotation" in d]
        rotation = int(rotations[0]) if rotations else int(video.get("tags", {}).get("rotate", 0))
        return MediaInfo(
            duration=duration,
            width=width,
            height=height,
            has_audio=audio,
            rotation=rotation,
            video_codec=str(video.get("codec_name", "")),
            audio_only=not bool(videos),
        )
    except (ValueError, TypeError, KeyError, OverflowError, json.JSONDecodeError) as exc:
        if isinstance(exc, MediaError):
            raise
        raise MediaError("ffprobe returned malformed or unsupported metadata") from exc
