"""No model, GPU, network or media required for vector correctness tests."""

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from vsearch.index.vectors import IndexMismatch, SQLiteVectorStore, normalize
from vsearch.retrieve.moments import merge_hits


def test_vector_rank_and_version_guard(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path)
    store.add(
        "visual",
        "CLIP-v1",
        [
            ("a", "v1", 1.0, np.array([1.0, 0.0]), {"thumbnail": "a.jpg"}),
            ("b", "v2", 2.0, np.array([0.0, 1.0]), {}),
        ],
    )
    hits = store.search("visual", "CLIP-v1", np.array([1.0, 0.0]))
    assert [h.item_id for h in hits] == ["a", "b"]
    assert hits[0].score == pytest.approx(1)
    with pytest.raises(IndexMismatch):
        store.search("visual", "CLIP-v2", np.array([1.0, 0.0]))
    with pytest.raises(IndexMismatch):
        store.add("visual", "CLIP-v1", [("c", "v3", 1.0, np.ones(3), {})])
    assert store.delete_video("v1") == 1
    assert store.count("visual") == 1
    assert store.verify({"v2"}) == []
    assert store.verify(set()) == ["orphan-vector:v2"]


@pytest.mark.parametrize("values", [[0.0, 0.0], [float("nan"), 1.0], [], [[1.0, 2.0]]])
def test_bad_embeddings_rejected(values: list[float]) -> None:
    with pytest.raises(ValueError):
        normalize(np.asarray(values))


@given(
    st.lists(
        st.floats(-100, 100, allow_nan=False, allow_infinity=False), min_size=2, max_size=64
    ).filter(lambda v: any(abs(x) > 1e-6 for x in v))
)
def test_normalization_unit_length(values: list[float]) -> None:
    vector = normalize(np.array(values))
    assert float(np.linalg.norm(vector)) == pytest.approx(1, abs=1e-5)


def test_moment_merge_by_video(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path)
    store.add(
        "visual",
        "mock",
        [
            ("a", "v1", 2.0, np.array([1.0, 0.0]), {"thumbnail": "a.jpg"}),
            ("b", "v1", 3.0, np.array([0.8, 0.2]), {}),
            ("c", "v1", 20.0, np.array([0.7, 0.3]), {}),
            ("d", "v2", 2.0, np.array([1.0, 0.0]), {}),
        ],
    )
    moments = merge_hits(store.search("visual", "mock", np.array([1.0, 0.0])))
    assert len(moments) == 3
    assert any(m.frame_ids == ("a", "b") for m in moments)
    assert all(m.end > m.start for m in moments)


def test_empty_index_and_limit(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path)
    assert store.search("visual", "not-created", np.ones(2)) == []
    assert store.search("visual", "not-created", np.ones(2), limit=0) == []
