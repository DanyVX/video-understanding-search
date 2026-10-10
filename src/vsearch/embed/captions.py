"""Cost-capped, opt-in scene caption interface with content hash cache."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class Captioner(Protocol):
    model_id: str
    def caption(self, frame: Path, prompt: str) -> str: ...


@dataclass(frozen=True)
class CaptionPlan:
    frames: int
    cost_per_frame_usd: float
    budget_usd: float
    timeout_seconds: float = 10.0

    def validate(self) -> None:
        if self.frames < 0 or not all(math.isfinite(x) and x >= 0 for x in (
            self.cost_per_frame_usd, self.budget_usd,
        )):
            raise ValueError("Invalid caption budget")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Invalid caption timeout")

    def estimate(self) -> float:
        self.validate()
        return self.frames * self.cost_per_frame_usd

    def allowed(self) -> bool:
        return self.estimate() <= self.budget_usd and self.budget_usd > 0


def cache_key(frame: bytes, model_id: str, prompt_version: str) -> str:
    if not model_id.strip() or not prompt_version.strip():
        raise ValueError("Model and prompt version required")
    digest = hashlib.sha256()
    for part in (frame, model_id.encode(), prompt_version.encode()):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


class CaptionCache:
    """Atomic JSON files, scoped to a specific model/prompt/frame revision."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def get(self, key: str) -> str | None:
        if len(key) != 64 or not all(c in "0123456789abcdef" for c in key):
            raise ValueError("Unsafe cache key")
        path = self.root / f"{key}.json"
        if not path.exists():
            return None
        loaded = json.loads(path.read_text(encoding="utf-8"))
        value = loaded.get("caption")
        return value if isinstance(value, str) and value.strip() else None

    def put(self, key: str, caption: str) -> None:
        if not caption.strip():
            raise ValueError("Refusing empty caption")
        if len(key) != 64 or not all(c in "0123456789abcdef" for c in key):
            raise ValueError("Unsafe cache key")
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"{key}.json"
        staged = target.with_suffix(".part")
        try:
            staged.write_text(json.dumps({"caption": caption}, ensure_ascii=False),
                              encoding="utf-8")
            staged.replace(target)
        finally:
            staged.unlink(missing_ok=True)


def generate_captions(
    frames: list[Path],
    provider: Captioner,
    cache: CaptionCache,
    plan: CaptionPlan,
    *,
    prompt: str = "Describe only visible details, without speculation.",
    prompt_version: str = "v1",
    dry_run: bool = True,
) -> list[str]:
    """Refuse unbudgeted API calls. Cache hits do not spend the budget.

    Provider itself must enforce network timeout; this boundary cannot interrupt
    blocked third-party SDK calls. Caller must verify provider billing.
    """
    if len(frames) != plan.frames:
        raise ValueError("Frame count differs from cost plan")
    if not dry_run and not plan.allowed():
        raise ValueError("Captioning blocked: budget is missing or exceeded")
    if dry_run:
        return []
    result: list[str] = []
    for frame in frames:
        key = cache_key(frame.read_bytes(), provider.model_id, prompt_version)
        cached = cache.get(key)
        if cached is not None:
            result.append(cached)
            continue
        generated = provider.caption(frame, prompt).strip()
        if not generated:
            raise ValueError("Caption provider returned an empty response")
        cache.put(key, generated)
        result.append(generated)
    return result
