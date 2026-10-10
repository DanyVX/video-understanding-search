"""Cross-store deletion/repair checks, no real videos or network needed."""

from pathlib import Path

import numpy as np

from vsearch.index.integrity import prune_orphan_vectors, verify_all
from vsearch.index.manifest import Manifest
from vsearch.index.vectors import SQLiteVectorStore


def test_verify_and_repair_missing_video(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path)
    store.add("visual", "test-model", [
        ("frame1", "missing-video", 2, np.array([1.0, 0.0]), {})
    ])
    issues = verify_all(tmp_path)
    assert any("orphan-vector" in issue for issue in issues)
    assert prune_orphan_vectors(tmp_path) == 1
    assert verify_all(tmp_path) == []


def test_end_to_end_deleted_video_has_no_vectors(tmp_path: Path) -> None:
    db = Manifest(tmp_path)
    video = "a" * 24
    db.upsert_video(video, "clip", None, video, 5, 20, 20, False, "v1")
    db.set_status(video, "done")
    vectors = SQLiteVectorStore(tmp_path)
    vectors.add("caption_dense", "fake", [
        ("doc0", video, 1, np.array([1.0, 0.0]), {})
    ])
    assert db.erase(video)
    assert verify_all(tmp_path) == [f"orphan-vector:{video}"]
    assert prune_orphan_vectors(tmp_path) == 1
    assert verify_all(tmp_path) == []


def test_dimension_corruption_is_reported(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path)
    store.add("caption_dense", "fake", [
        ("x", "missing", 1, np.array([0.0, 1.0]), {})
    ])
    with store._connect() as conn:
        conn.execute("UPDATE vectors SET embedding=? WHERE item_id='x'", (b"x",))
    assert any("bad-vector-dimensions" in issue for issue in verify_all(tmp_path))
