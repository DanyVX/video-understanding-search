"""Bounded in-process ingest jobs accepting only local staged media filenames."""

from __future__ import annotations

import secrets
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Semaphore
from typing import Literal

from vsearch.index.manifest import Manifest
from vsearch.ingest.pipeline import IngestOptions, ingest

Status = Literal["queued", "processing", "done", "failed"]


class JobError(ValueError):
    """Invalid or unsupported ingest job request."""


@dataclass(frozen=True)
class Job:
    id: str
    status: Status
    video_id: str | None = None
    error: str | None = None


def safe_input(root: Path, filename: str, max_bytes: int) -> Path:
    """Accept a basename, never a user-supplied absolute or parent path."""
    if (
        not filename
        or len(filename) > 255
        or filename in (".", "..")
        or "/" in filename
        or "\\" in filename
        or Path(filename).name != filename
    ):
        raise JobError("Invalid media filename")
    incoming = (root / "incoming").resolve()
    candidate = (incoming / filename).resolve()
    if not candidate.is_relative_to(incoming) or not candidate.is_file():
        raise JobError("Staged media file not found")
    if not (0 < candidate.stat().st_size <= max_bytes):
        raise JobError("Media file is empty or exceeds configured size limit")
    with candidate.open("rb") as stream:
        head = stream.read(16)
    # Deliberately narrow supported container magic types. ffprobe is still authoritative.
    iso = len(head) >= 8 and head[4:8] == b"ftyp"
    matroska = head.startswith(b"\x1a\x45\xdf\xa3")
    avi = head[:4] == b"RIFF" and head[8:12] == b"AVI "
    if not (iso or matroska or avi):
        raise JobError("Unsupported or unrecognized media container")
    return candidate


class IngestJobs:
    """Per-process job state; manifest checkpoints enable restart after process loss."""

    def __init__(self, workers: int = 2) -> None:
        if workers < 1:
            raise ValueError("workers must be positive")
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vsearch")
        self._slots = Semaphore(workers)
        self._mutex = Lock()
        self._jobs: dict[str, Job] = {}

    def get(self, job_id: str) -> Job | None:
        with self._mutex:
            return self._jobs.get(job_id)

    def submit(
        self, root: Path, filename: str, max_bytes: int, *,
        keep_source: bool = False,
    ) -> Job:
        path = safe_input(root, filename, max_bytes)
        if not self._slots.acquire(blocking=False):
            raise JobError("Ingest capacity reached")
        job_id = secrets.token_hex(12)
        entry = Job(job_id, "queued")
        with self._mutex:
            self._jobs[job_id] = entry
        try:
            self._pool.submit(self._run, entry.id, root, path, keep_source)
        except BaseException:
            with self._mutex:
                del self._jobs[job_id]
            self._slots.release()
            raise
        return entry

    def _run(self, job_id: str, root: Path, path: Path, keep_source: bool) -> None:
        try:
            with self._mutex:
                self._jobs[job_id] = Job(job_id, "processing")
            result = ingest(path, Manifest(root), IngestOptions(keep_source=keep_source))
            with self._mutex:
                self._jobs[job_id] = Job(job_id, "done", video_id=str(result["id"]))
        except Exception as exc:
            # Never leak raw filenames, media contents or tool stderr to clients.
            with self._mutex:
                self._jobs[job_id] = Job(job_id, "failed", error=type(exc).__name__)
        finally:
            self._slots.release()


JOBS = IngestJobs()
