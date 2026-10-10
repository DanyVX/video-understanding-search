"""Cross-platform developer commands; no bash required."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMANDS: dict[str, list[list[str]]] = {
    "lint": [
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "ruff", "format", "--diff", "."],
        [sys.executable, "-m", "ruff", "format", "--check", "."],
    ],
    "format": [
        [sys.executable, "-m", "ruff", "check", "--fix", "."],
        [sys.executable, "-m", "ruff", "format", "."],
    ],
    "typecheck": [[sys.executable, "-m", "mypy", "src"]],
    "test": [
        [sys.executable, "-m", "pytest", "--cov=vsearch", "--cov-branch",
         "--cov-report=term-missing", "--cov-fail-under=70", "-m", "not slow and not gpu"]
    ],
    "test-fast": [[sys.executable, "-m", "pytest", "-q", "-m", "not slow and not gpu"]],
    "bench": [[sys.executable, "-m", "vsearch", "doctor", "--json"]],
    "run": [[sys.executable, "-m", "vsearch", "doctor"]],
}


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/tasks.py <setup|lint|format|typecheck|test|test-fast|"
              "bench|run|docker-build|clean>", file=sys.stderr)
        return 2
    task = sys.argv[1]
    if task == "setup":
        if shutil.which("uv"):
            return subprocess.call(["uv", "sync", "--extra", "dev"], cwd=ROOT)
        return subprocess.call(
            [sys.executable, "-m", "pip", "install", "-e", ".[dev]"], cwd=ROOT
        )
    if task == "docker-build":
        print("Docker image is not implemented. M6 will add a reviewed Dockerfile.")
        return 2
    if task == "clean":
        for name in (".pytest_cache", ".mypy_cache", ".ruff_cache", "htmlcov"):
            shutil.rmtree(ROOT / name, ignore_errors=True)
        for name in (".coverage", "coverage.xml"):
            (ROOT / name).unlink(missing_ok=True)
        return 0
    if task not in COMMANDS:
        print(f"Unknown task: {task}", file=sys.stderr)
        return 2
    for command in COMMANDS[task]:
        status = subprocess.call(command, cwd=ROOT)
        if status:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
