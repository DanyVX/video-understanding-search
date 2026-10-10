"""Caption defaults never invoke a provider or spend money."""

from pathlib import Path

import pytest

from vsearch.embed.captions import CaptionCache, CaptionPlan, cache_key, generate_captions


class FakeCaptioner:
    model_id = "fake-model-v1"

    def __init__(self) -> None:
        self.calls = 0

    def caption(self, frame: Path, prompt: str) -> str:
        self.calls += 1
        return "a test frame"


def test_dry_run_never_calls_provider(tmp_path: Path) -> None:
    path = tmp_path / "frame.jpg"
    path.write_bytes(b"image")
    provider = FakeCaptioner()
    assert generate_captions([path], provider, CaptionCache(tmp_path / "cache"),
                             CaptionPlan(1, 1, 0)) == []
    assert provider.calls == 0


def test_budget_is_a_hard_gate(tmp_path: Path) -> None:
    path = tmp_path / "frame.jpg"
    path.write_bytes(b"image")
    with pytest.raises(ValueError, match="budget"):
        generate_captions([path], FakeCaptioner(), CaptionCache(tmp_path),
                          CaptionPlan(1, 0.1, 0), dry_run=False)


def test_cache_is_reproducible(tmp_path: Path) -> None:
    path = tmp_path / "frame.jpg"
    path.write_bytes(b"image")
    provider = FakeCaptioner()
    cache = CaptionCache(tmp_path / "cache")
    plan = CaptionPlan(1, 0.01, 1)
    assert generate_captions([path], provider, cache, plan, dry_run=False) == [
        "a test frame"
    ]
    assert generate_captions([path], provider, cache, plan, dry_run=False) == [
        "a test frame"
    ]
    assert provider.calls == 1
    assert cache_key(b"a", "m", "p") != cache_key(b"a", "m", "p2")


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        CaptionPlan(1, -1, 1).estimate()
    with pytest.raises(ValueError):
        cache_key(b"a", "", "v1")
