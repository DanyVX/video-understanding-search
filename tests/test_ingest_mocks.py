"""Exercise the production ingest state machine with deterministic decoder fixtures."""

import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from vsearch.index.manifest import Manifest
from vsearch.ingest import pipeline, probe


def fake_media(tmp_path: Path) -> Path:
    path = tmp_path / "fake media Ω.mp4"
    path.write_bytes(b"deterministic mock fixture")
    return path


def fake_info(duration: float = 5.0, audio_only: bool = False) -> probe.MediaInfo:
    return probe.MediaInfo(
        duration=duration, width=0 if audio_only else 60,
        height=0 if audio_only else 60,
        has_audio=audio_only, audio_only=audio_only,
    )


def fake_decode(argv: list[str], *, timeout: float = 120) -> subprocess.CompletedProcess[bytes]:
    output = Path(argv[-1]).parent
    Image.new("RGB", (60, 60), color=(120, 90, 50)).save(output / "frame_000001.jpg")
    Image.new("RGB", (60, 60), color=(60, 120, 80)).save(output / "frame_000002.jpg")
    return subprocess.CompletedProcess(argv, 0, b"", b"")


def test_resume_dedupe_verify_delete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = fake_media(tmp_path)
    db = Manifest(tmp_path / "state")
    monkeypatch.setattr(pipeline, "probe", lambda _p: fake_info())
    monkeypatch.setattr(pipeline, "run_media_command", fake_decode)
    config = pipeline.IngestOptions(fps=1, chunk_seconds=2)
    item = pipeline.ingest(path, db, config)
    assert item["status"] == "done"
    assert db.chunks_done(str(item["id"])) == {0, 1, 2}
    assert db.frames(str(item["id"]))
    assert db.verify() == []
    assert pipeline.ingest(path, db, config)["id"] == item["id"]
    assert pipeline.delete_video(db, str(item["id"]))
    assert not pipeline.delete_video(db, str(item["id"]))
    assert db.list_videos() == []


def test_failed_decode_has_resumable_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = fake_media(tmp_path)
    db = Manifest(tmp_path / "state")
    monkeypatch.setattr(pipeline, "probe", lambda _p: fake_info())
    attempts = {"count": 0}

    def unstable(argv: list[str], *, timeout: float = 120) -> subprocess.CompletedProcess[bytes]:
        attempts["count"] += 1
        if attempts["count"] == 2:
            return subprocess.CompletedProcess(argv, 1, b"", b"bad")
        return fake_decode(argv, timeout=timeout)

    monkeypatch.setattr(pipeline, "run_media_command", unstable)
    with pytest.raises(probe.MediaError, match="Decoding failed"):
        pipeline.ingest(path, db, pipeline.IngestOptions(fps=1, chunk_seconds=2))
    ident = pipeline.sha256_file(path)[:24]
    assert db.get_video(ident)["status"] == "failed"
    assert db.chunks_done(ident) == {0}
    assert db.frames(ident)
    monkeypatch.setattr(pipeline, "run_media_command", fake_decode)
    assert pipeline.ingest(path, db, pipeline.IngestOptions(fps=1, chunk_seconds=2))["status"] == "done"
    assert db.chunks_done(ident) == {0, 1, 2}


def test_audio_only_is_valid_without_visual_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = fake_media(tmp_path)
    db = Manifest(tmp_path / "state")
    monkeypatch.setattr(pipeline, "probe", lambda _p: fake_info(audio_only=True))
    item = pipeline.ingest(path, db, pipeline.IngestOptions(chunk_seconds=2))
    assert item["status"] == "done"
    assert item["has_audio"] == 1
    assert db.frames(str(item["id"])) == []


def test_safe_delete_rejects_paths(tmp_path: Path) -> None:
    db = Manifest(tmp_path)
    for bad in ("", "../root", "x" * 24, "/" * 24, "ab" * 13):
        assert not pipeline.delete_video(db, bad)


def test_probe_normal_and_audio_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = fake_media(tmp_path)
    monkeypatch.setattr(probe.shutil, "which", lambda _exe: "/mock/ffprobe")
    payload = {
        "format": {"duration": "4.25"},
        "streams": [
            {"codec_type": "video", "width": 640, "height": 360,
             "codec_name": "h264", "tags": {"rotate": "90"}},
            {"codec_type": "audio", "codec_name": "aac"},
        ],
    }
    monkeypatch.setattr(
        probe, "run_media_command",
        lambda *a, **k: subprocess.CompletedProcess([], 0, json.dumps(payload).encode(), b""),
    )
    info = probe.probe(path)
    assert info.duration == 4.25
    assert info.has_audio
    assert info.rotation == 90


def test_probe_rejects_malformed_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = fake_media(tmp_path)
    monkeypatch.setattr(probe.shutil, "which", lambda _exe: "/mock/ffprobe")
    monkeypatch.setattr(
        probe, "run_media_command",
        lambda *a, **k: subprocess.CompletedProcess([], 0, b"not-json", b""),
    )
    with pytest.raises(probe.MediaError, match="malformed"):
        probe.probe(path)


def test_command_timeout_and_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    def timed_out(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1)

    monkeypatch.setattr(probe.subprocess, "run", timed_out)
    with pytest.raises(probe.MediaError, match="timed out"):
        probe.run_media_command(["ffmpeg", "file.mp4"])


def test_hash_and_scene_helpers(tmp_path: Path) -> None:
    file = tmp_path / "frame.jpg"
    Image.new("RGB", (60, 60), color=(64, 64, 64)).save(file)
    hash_value, luminance = pipeline._perceptual_hash(file)
    assert len(hash_value) == 16
    assert luminance > 0
    assert pipeline._distance(hash_value, hash_value) == 0
