"""No media decode needed for HTTP job concurrency and path safety."""

from pathlib import Path
from threading import Event

import pytest

from vsearch.api import jobs


def test_traversal_and_magic_bytes(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    movie = incoming / "safe name.MP4"
    movie.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"0" * 50)
    assert jobs.safe_input(tmp_path, movie.name, 100).name == movie.name
    for path in ("../secret.mp4", "..\\secret.mp4", "/etc/passwd", ""):
        with pytest.raises(jobs.JobError):
            jobs.safe_input(tmp_path, path, 100)
    with pytest.raises(jobs.JobError):
        jobs.safe_input(tmp_path, movie.name, 16)


def test_unsupported_magic(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "bad.mp4").write_bytes(b"random" * 8)
    with pytest.raises(jobs.JobError, match="Unsupported"):
        jobs.safe_input(tmp_path, "bad.mp4", 1000)


def test_inprocess_job_completes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "clip.mp4").write_bytes(b"\0\0\0\0ftypmore" + b"0" * 30)
    running = Event()
    finished = Event()

    def mocked_ingest(path: Path, manifest: object, opts: object) -> dict[str, str]:
        running.set()
        finished.wait(timeout=2)
        return {"id": "a" * 24}

    monkeypatch.setattr(jobs, "ingest", mocked_ingest)
    manager = jobs.IngestJobs(workers=1)
    first = manager.submit(tmp_path, "clip.mp4", 1000)
    assert running.wait(timeout=3)
    with pytest.raises(jobs.JobError, match="capacity"):
        manager.submit(tmp_path, "clip.mp4", 1000)
    finished.set()
    from time import monotonic, sleep

    deadline = monotonic() + 3
    while manager.get(first.id).status != "done" and monotonic() < deadline:
        sleep(0.005)
    completed = manager.get(first.id)
    assert completed is not None and completed.video_id == "a" * 24
