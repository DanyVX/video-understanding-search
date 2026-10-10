"""Reproducible, privacy-limited runtime metadata for diagnostics and benchmarks."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PACKAGES = (
    "pydantic-settings",
    "typer",
    "numpy",
    "faiss-cpu",
    "open-clip-torch",
    "faster-whisper",
    "scenedetect",
)


def _output(argv: list[str], timeout: float = 3.0) -> str | None:
    try:
        result = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    return result.stdout.strip() or None


def _total_ram_mb() -> int | None:
    """Report physical RAM on POSIX; return unknown where unsupported."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if isinstance(pages, int) and isinstance(page_size, int):
            return pages * page_size // (1024 * 1024)
    except (AttributeError, OSError, ValueError):
        pass
    return None


def _network_status() -> str:
    try:
        with socket.create_connection(("pypi.org", 443), timeout=2):
            return "reachable"
    except OSError:
        return "unreachable"


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _git_commit() -> str:
    return _output(["git", "rev-parse", "--verify", "HEAD"]) or "unknown"


def collect_environment(*, check_network: bool = False) -> dict[str, Any]:
    """Capture environment metadata without query text, file names or secrets.

    A benchmark should write this metadata alongside its measured values.
    Network probing is disabled by default.
    """
    disk = shutil.disk_usage(Path.home())
    gpu = _output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
    )
    cuda_details = _output(["nvcc", "--version"])
    ffmpeg = _output(["ffmpeg", "-version"])
    ffprobe = _output(["ffprobe", "-version"])

    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "os": platform.platform(),
        "shell": Path(os.getenv("SHELL") or os.getenv("COMSPEC") or "unknown").name,
        "python": platform.python_version(),
        "python_executable_name": Path(sys.executable).name,
        "cpu_count": os.cpu_count(),
        "ram_mb": _total_ram_mb(),
        "free_disk_mb": disk.free // (1024 * 1024),
        "gpu": gpu or "none-detected",
        "cuda": cuda_details.splitlines()[-1] if cuda_details else "none-detected",
        "uv_available": shutil.which("uv") is not None,
        "pip_available": shutil.which("pip") is not None,
        "docker_available": shutil.which("docker") is not None,
        "ffmpeg": ffmpeg.splitlines()[0] if ffmpeg else "unavailable",
        "ffprobe": ffprobe.splitlines()[0] if ffprobe else "unavailable",
        "git_identity_configured": bool(_output(["git", "config", "--get", "user.name"]))
        and bool(_output(["git", "config", "--get", "user.email"])),
        "git_commit": _git_commit(),
        "internet": _network_status() if check_network else "not-checked",
        "library_versions": _package_versions(),
    }
