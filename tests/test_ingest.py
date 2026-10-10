"""Deterministic local pipeline tests with optional system ffmpeg smoke coverage."""

import shutil
import subprocess
from pathlib import Path

import pytest

from vsearch.index.manifest import Manifest
from vsearch.ingest.pipeline import IngestOptions, delete_video, ingest
from vsearch.ingest.probe import MediaError, probe, sha256_file


def test_options_limits() -> None:
    for fps, chunk in [(0.01, 120), (8.0, 120), (1.0, 1), (4, 600)]:
        with pytest.raises(ValueError):
            IngestOptions(fps=fps, chunk_seconds=chunk).validate()


def test_manifest_atomic_commit_and_resume(tmp_path: Path) -> None:
    db = Manifest(tmp_path)
    db.upsert_video("abc", "some file.mp4", None, "digest", 10, 320, 180, True, "v1")
    db.set_status("abc", "processing")
    db.begin_chunk("abc", 0)
    assert db.chunks_done("abc") == set()
    db.finish_chunk("abc", 0, [("frame1", 1.0, 0, "abcd", "thumbs/abc/frame1.jpg")])
    assert db.chunks_done("abc") == {0}
    assert len(db.verify()) == 1
    assert len(db.repair()) == 1
    assert db.get_video("abc")["status"] == "failed"
    assert db.erase("abc")
    assert db.frames("abc") == []


def test_invalid_file(tmp_path: Path) -> None:
    p = tmp_path / "bad.mp4"
    p.write_bytes(b"not-video")
    if shutil.which("ffprobe"):
        with pytest.raises(MediaError):
            probe(p)


def test_sha256_unicode_filename(tmp_path: Path) -> None:
    p = tmp_path / "دانش video name.bin"
    p.write_bytes(b"test")
    assert sha256_file(p) == sha256_file(p)


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg missing"
)
def test_synthetic_video_ingest_dedupe_and_delete(tmp_path: Path) -> None:
    movie = tmp_path / "synthetic video Ω.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x96:rate=2",
            "-t",
            "2.5",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(movie),
        ],
        check=True,
        timeout=30,
    )
    manifest = Manifest(tmp_path / "state")
    first = ingest(movie, manifest, IngestOptions(fps=1, chunk_seconds=2))
    assert first["status"] == "done"
    assert len(manifest.frames(str(first["id"]))) >= 1
    second = ingest(movie, manifest, IngestOptions(fps=1, chunk_seconds=2))
    assert first["id"] == second["id"]
    assert manifest.verify() == []
    assert delete_video(manifest, str(first["id"]))
    assert manifest.list_videos() == []
