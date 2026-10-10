"""Offline text and mixed-channel search using explicit mocked vector models."""

from pathlib import Path

from vsearch.asr.transcript import Segment
from vsearch.index.manifest import Manifest
from vsearch.index.transcripts import TranscriptIndex
from vsearch.retrieve.service import search_moments


def test_text_only_does_not_download_weights(tmp_path: Path) -> None:
    video = "a" * 24
    store = Manifest(tmp_path)
    store.upsert_video(video, "clip.mp4", None, video, 10, 200, 100, True, "m1")
    store.set_status(video, "done")
    TranscriptIndex(tmp_path).replace(video, [Segment(3, 5, "the spaceship launches")])
    results = search_moments(tmp_path, "spaceship", limit=5)
    assert len(results) == 1
    assert results[0].video_id == video
    assert results[0].start == 3
    assert results[0].channels["transcript"] > 0


def test_empty_index_and_queries(tmp_path: Path) -> None:
    assert search_moments(tmp_path, "unanswerable") == []
    assert search_moments(tmp_path, "unanswerable", include_visual=True) == []
