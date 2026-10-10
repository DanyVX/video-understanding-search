"""Rank-based multimodal fusion with explicit absent-channel handling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Hit:
    video_id: str
    start: float
    end: float
    score: float
    channel: str
    snippet: str = ""

    def __post_init__(self) -> None:
        if (not self.video_id or not self.channel or
                not all(math.isfinite(v) for v in (self.start, self.end, self.score)) or
                self.start < 0 or self.end < self.start):
            raise ValueError("Invalid retrieval hit")


@dataclass(frozen=True)
class Moment:
    video_id: str
    start: float
    end: float
    score: float
    channels: dict[str, float]
    snippet: str


def reciprocal_rank_fusion(
    channels: Mapping[str, Sequence[Hit]],
    *,
    weights: Mapping[str, float] | None = None,
    k: int = 60,
    round_seconds: float = 2.0,
) -> list[Moment]:
    """Fuse aligned time bins using normalized weighted reciprocal ranks.

    Channels with no hits are omitted from the denominator so an audio-free
    video is not penalized by an absent speech channel. Raw scores never
    participate in RRF: CLIP cosine and BM25 are on incompatible scales.
    """
    if k < 1 or round_seconds <= 0 or not math.isfinite(round_seconds):
        raise ValueError("Invalid RRF parameters")
    chosen = dict(weights or {})
    if any(not math.isfinite(v) or v < 0 for v in chosen.values()):
        raise ValueError("Channel weights must be nonnegative finite numbers")
    active = {name: hits for name, hits in channels.items() if hits}
    denom = sum(chosen.get(name, 1.0) for name in active)
    if denom <= 0:
        return []
    moments: dict[tuple[str, int], Moment] = {}
    for name, hits in active.items():
        weight = chosen.get(name, 1.0) / denom
        ranked = sorted(hits, key=lambda hit: (
            -hit.score, hit.video_id, hit.start, hit.end
        ))
        seen: set[tuple[str, int]] = set()
        for rank, hit in enumerate(ranked, start=1):
            bucket = math.floor((hit.start + hit.end) / (2 * round_seconds))
            key = (hit.video_id, bucket)
            if key in seen:
                continue
            seen.add(key)
            amount = weight / (k + rank)
            current = moments.get(key)
            if current is None:
                moments[key] = Moment(hit.video_id, hit.start, hit.end, amount,
                                      {name: amount}, hit.snippet)
            else:
                details = dict(current.channels)
                details[name] = details.get(name, 0.0) + amount
                moments[key] = Moment(
                    hit.video_id, min(current.start, hit.start),
                    max(current.end, hit.end), current.score + amount,
                    details, current.snippet or hit.snippet,
                )
    return sorted(moments.values(), key=lambda item: (
        -item.score, item.video_id, item.start,
    ))


def merge_moments(
    moments: Sequence[Moment], *,
    gap_tolerance: float = 1.0,
    min_seconds: float = 0.25,
    limit: int = 20,
) -> list[Moment]:
    """Merge adjacent same-video results and cap generic-query result floods."""
    if (not math.isfinite(gap_tolerance) or gap_tolerance < 0 or
            not math.isfinite(min_seconds) or min_seconds <= 0 or limit < 1):
        raise ValueError("Invalid merging configuration")
    ordered = sorted(moments, key=lambda m: (m.video_id, m.start, m.end))
    merged: list[Moment] = []
    for item in ordered:
        if merged and merged[-1].video_id == item.video_id and (
            item.start <= merged[-1].end + gap_tolerance
        ):
            previous = merged.pop()
            channel_scores = dict(previous.channels)
            for name, score in item.channels.items():
                channel_scores[name] = max(channel_scores.get(name, 0), score)
            merged.append(Moment(
                item.video_id, previous.start, max(item.end, previous.end),
                max(item.score, previous.score), channel_scores,
                previous.snippet or item.snippet,
            ))
        else:
            merged.append(item)
    return sorted(
        [Moment(m.video_id, m.start, max(m.end, m.start + min_seconds), m.score,
                m.channels, m.snippet) for m in merged],
        key=lambda m: (-m.score, m.video_id, m.start),
    )[:limit]
