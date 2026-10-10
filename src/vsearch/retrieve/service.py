"""Search orchestrator: text BM25 plus optional CLIP frame vectors, fused by rank."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

from vsearch.index.transcripts import TranscriptIndex
from vsearch.retrieve.fusion import Hit, Moment, merge_moments, reciprocal_rank_fusion


@lru_cache(maxsize=1)
def _visual_model() -> object:
    """Keep one model in RAM instead of reloading per HTTP request."""
    from vsearch.embed.visual import OpenClipEncoder

    return OpenClipEncoder()


def search_moments(
    root: Path,
    query: str,
    *,
    limit: int = 10,
    include_visual: bool = False,
    weights: dict[str, float] | None = None,
) -> list[Moment]:
    """Return time-bounded moments. No ML model is loaded for text-only searches."""
    if not query.strip() or len(query) > 512:
        raise ValueError("Query must contain 1-512 non-whitespace characters")
    if not 1 <= limit <= 50:
        raise ValueError("Result limit must be between 1 and 50")
    transcript_hits = TranscriptIndex(root).search(query, limit=min(200, limit * 4))
    channels: dict[str, list[Hit]] = {
        "transcript": [
            Hit(
                result.video_id,
                result.start,
                result.end,
                result.score,
                "transcript",
                result.snippet,
            )
            for result in transcript_hits
        ]
    }
    if include_visual:
        from vsearch.embed.visual import VisualEncoder
        from vsearch.index.vectors import SQLiteVectorStore

        store = SQLiteVectorStore(root)
        if store.count("visual"):
            encoder: VisualEncoder = _visual_model()  # type: ignore[assignment]
            vectors = store.search(
                "visual", encoder.model_key, encoder.encode_text(query), limit=min(200, limit * 4)
            )
            channels["visual"] = [
                Hit(
                    item.video_id,
                    max(0, item.t - 0.5),
                    item.t + 0.5,
                    item.score,
                    "visual",
                )
                for item in vectors
                if math.isfinite(item.score)
            ]
    ranked = reciprocal_rank_fusion(channels, weights=weights)
    return merge_moments(ranked, gap_tolerance=2.0, min_seconds=0.5, limit=limit)
