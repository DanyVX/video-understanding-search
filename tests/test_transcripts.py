"""Deterministic, offline ASR indexing tests."""

from pathlib import Path

import pytest

from vsearch.asr.transcript import Segment, filter_segments, windows
from vsearch.index.manifest import Manifest
from vsearch.index.transcripts import TranscriptIndex, tokenize


def test_filter_silence_and_repetition() -> None:
    data = [
        Segment(0, 1, "hello", 0),
        Segment(1, 2, "hello", 0),
        Segment(2, 3, "hello", 0),
        Segment(3, 4, "fabricated", 0.9),
        Segment(4, 5, "", 0),
    ]
    assert [x.text for x in filter_segments(data)] == ["hello", "hello"]


def test_windows_keep_original_boundaries() -> None:
    pieces = [
        Segment(0.1, 1.8, "open"),
        Segment(2.0, 3.0, "laptop"),
        Segment(12.0, 14.0, "later"),
    ]
    assert windows(pieces, max_seconds=5) == [
        Segment(0.1, 3.0, "open laptop"),
        Segment(12.0, 14.0, "later"),
    ]


def test_invalid_filter() -> None:
    with pytest.raises(ValueError):
        filter_segments([], max_no_speech=-1)


def test_tokenize_unicode() -> None:
    assert tokenize("Hello, دنیا!") == ["hello", "دنیا"]


def test_rank_and_delete(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path)
    for identifier in ("a" * 24, "b" * 24):
        manifest.upsert_video(identifier, "test.mp4", None, identifier, 10, 100, 50,
                              True, "v1")
        manifest.set_status(identifier, "done")
    index = TranscriptIndex(tmp_path)
    index.replace("a" * 24, [Segment(1, 3, "someone opens a laptop")])
    index.replace("b" * 24, [Segment(5, 7, "the doors are closed")])
    hits = index.search("opens laptop")
    assert hits[0].video_id == "a" * 24
    assert hits[0].start == 1
    assert len(hits) == 1
    manifest.erase("a" * 24)
    assert index.search("opens laptop") == []


def test_bad_video_and_invalid_segment(tmp_path: Path) -> None:
    index = TranscriptIndex(tmp_path)
    with pytest.raises(ValueError, match="Unknown"):
        index.replace("missing", [Segment(0, 1, "test")])
    assert index.search("   ") == []
