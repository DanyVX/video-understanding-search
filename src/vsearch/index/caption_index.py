"""Timestamped caption BM25 over a separate SQLite channel."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from vsearch.asr.transcript import Segment
from vsearch.index.manifest import Manifest
from vsearch.index.transcripts import TextHit, tokenize


class CaptionIndex:
    """Caption records are plain data, not executable prompt instructions."""

    def __init__(self, root: Path) -> None:
        self.manifest = Manifest(root)
        with self.manifest.connection() as conn:
            conn.executescript(
                """CREATE TABLE IF NOT EXISTS caption_segments(
                video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
                start REAL NOT NULL CHECK(start>=0),
                end REAL NOT NULL CHECK(end>=start),
                caption TEXT NOT NULL,
                model_key TEXT NOT NULL,
                PRIMARY KEY(video_id,start,end,model_key)
                );
                CREATE INDEX IF NOT EXISTS captions_video_time
                ON caption_segments(video_id,start);"""
            )

    def replace(
        self, video_id: str, segments: list[Segment], *,
        model_key: str = "user-provided-v1",
    ) -> None:
        if not model_key.strip() or len(model_key) > 200:
            raise ValueError("Invalid caption model key")
        if self.manifest.get_video(video_id) is None:
            raise ValueError("Unknown video ID")
        if any(not segment.valid() for segment in segments):
            raise ValueError("Invalid caption timestamp/text")
        with self.manifest.connection() as conn:
            conn.execute(
                "DELETE FROM caption_segments WHERE video_id=? AND model_key=?",
                (video_id, model_key),
            )
            conn.executemany(
                """INSERT INTO caption_segments(video_id,start,end,caption,model_key)
                   VALUES(?,?,?,?,?)""",
                [(video_id, s.start, s.end, s.text, model_key) for s in segments],
            )

    def search(self, query: str, limit: int = 10) -> list[TextHit]:
        terms = tokenize(query)
        if not terms or limit < 1:
            return []
        with self.manifest.connection() as conn:
            rows = conn.execute(
                """SELECT c.video_id,c.start,c.end,c.caption FROM caption_segments c
                   JOIN videos v ON v.id=c.video_id WHERE v.status='done'
                   ORDER BY c.video_id,c.start"""
            ).fetchall()
        docs = [tokenize(str(row["caption"])) for row in rows]
        if not docs:
            return []
        length = sum(len(doc) for doc in docs) / len(docs)
        freq = Counter(token for doc in docs for token in set(doc))
        found: list[TextHit] = []
        for row, words in zip(rows, docs, strict=True):
            tf = Counter(words)
            score = 0.0
            for term in set(terms):
                n = freq[term]
                if n == 0:
                    continue
                idf = math.log(1 + (len(docs) - n + 0.5) / (n + 0.5))
                normalization = tf[term] + 1.2 * (
                    0.25 + 0.75 * len(words) / max(length, 1)
                )
                score += idf * tf[term] * 2.2 / max(normalization, 1e-9)
            if score > 0:
                found.append(TextHit(
                    str(row["video_id"]), float(row["start"]), float(row["end"]),
                    score, str(row["caption"])
                ))
        return sorted(found, key=lambda hit: (-hit.score, hit.video_id, hit.start))[:limit]
