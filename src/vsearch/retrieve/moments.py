"""Deterministic temporal clustering of timestamped retrieval hits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from vsearch.index.vectors import VectorHit


@dataclass(frozen=True)
class Moment:
    video_id: str
    start: float
    end: float
    score: float
    frame_ids: tuple[str, ...]
    thumbnail: str | None


def merge_hits(
    hits: list[VectorHit],
    *,
    gap_seconds: float = 3,
    min_length: float = 1,
    aggregation: Literal["max", "mean", "top2"] = "max",
) -> list[Moment]:
    if gap_seconds < 0 or min_length <= 0:
        raise ValueError("Gap and duration must be valid")
    if aggregation not in ("max", "mean", "top2"):
        raise ValueError("Unsupported score aggregation")
    by_video: dict[str, list[VectorHit]] = {}
    for hit in hits:
        by_video.setdefault(hit.video_id, []).append(hit)
    merged: list[Moment] = []

    def emit(video_id: str, cluster: list[VectorHit]) -> None:
        scores = sorted((x.score for x in cluster), reverse=True)
        score = (
            scores[0] if aggregation == "max"
            else sum(scores) / len(scores) if aggregation == "mean"
            else sum(scores[:2]) / min(2, len(scores))
        )
        start = max(0, cluster[0].t - min_length / 2)
        end = max(start + min_length, cluster[-1].t + min_length / 2)
        best = max(cluster, key=lambda x: (x.score, -x.t))
        thumbnail = best.payload.get("thumbnail")
        merged.append(Moment(
            video_id, round(start, 3), round(end, 3), float(score),
            tuple(x.item_id for x in cluster),
            str(thumbnail) if thumbnail is not None else None,
        ))

    for video_id, items in by_video.items():
        ordered = sorted(items, key=lambda item: (item.t, item.item_id))
        cluster = [ordered[0]]
        for item in ordered[1:]:
            if item.t - cluster[-1].t <= gap_seconds:
                cluster.append(item)
            else:
                emit(video_id, cluster)
                cluster = [item]
        emit(video_id, cluster)
    return sorted(merged, key=lambda moment: (-moment.score, moment.video_id, moment.start))


def search_visual(
    vectors: object, encoder: object, query: str, *, k: int = 20
) -> list[Moment]:
    """Search and cluster frames; callers may clamp moments to video duration."""
    from vsearch.embed.visual import VisualEncoder
    from vsearch.index.vectors import VectorStore

    if not query.strip():
        return []
    store: VectorStore = vectors  # type: ignore[assignment]
    model: VisualEncoder = encoder  # type: ignore[assignment]
    embeddings = model.encode_text(query)
    hits = store.search("visual", model.model_key, embeddings, limit=k)
    return merge_hits(hits)
