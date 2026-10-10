"""SQLite transcript records and dependency-free BM25 ranking for small corpora."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from vsearch.asr.transcript import Segment
from vsearch.index.manifest import Manifest

WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return WORD.findall(text.casefold())


@dataclass(frozen=True)
class TextHit:
    video_id: str
    start: float
    end: float
    score: float
    snippet: str


class TranscriptIndex:
    def __init__(self, root: Path) -> None:
        self.manifest = Manifest(root)
        with self.manifest.connection() as conn:
            conn.executescript(
                """CREATE TABLE IF NOT EXISTS transcript_segments (
                   video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
                   start REAL NOT NULL, end REAL NOT NULL, text TEXT NOT NULL,
                   PRIMARY KEY(video_id,start,end)
                   );
                   CREATE INDEX IF NOT EXISTS transcript_video_time
                   ON transcript_segments(video_id,start);"""
            )

    def replace(self, video_id: str, segments: list[Segment]) -> None:
        if self.manifest.get_video(video_id) is None:
            raise ValueError("Unknown video ID")
        if any(not segment.valid() for segment in segments):
            raise ValueError("Invalid ASR segment")
        with self.manifest.connection() as conn:
            conn.execute("DELETE FROM transcript_segments WHERE video_id=?", (video_id,))
            conn.executemany(
                "INSERT INTO transcript_segments(video_id,start,end,text) VALUES (?,?,?,?)",
                [(video_id, segment.start, segment.end, segment.text) for segment in segments],
            )

    def search(self, query: str, limit: int = 10) -> list[TextHit]:
        terms = tokenize(query)
        if not terms or limit < 1:
            return []
        with self.manifest.connection() as conn:
            rows = conn.execute(
                """SELECT t.video_id,t.start,t.end,t.text FROM transcript_segments t
                   INNER JOIN videos v ON v.id=t.video_id WHERE v.status='done'"""
            ).fetchall()
        docs = [tokenize(str(row["text"])) for row in rows]
        if not docs:
            return []
        n = len(docs)
        avg_length = sum(len(doc) for doc in docs) / n
        df = Counter(term for doc in docs for term in set(doc))
        results: list[TextHit] = []
        for row, doc in zip(rows, docs, strict=True):
            tf = Counter(doc)
            score = 0.0
            for term in set(terms):
                if not df[term]:
                    continue
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                denominator = tf[term] + 1.2 * (
                    1 - 0.75 + 0.75 * len(doc) / max(avg_length, 1)
                )
                score += idf * tf[term] * 2.2 / max(denominator, 1e-9)
            if score > 0:
                results.append(TextHit(
                    str(row["video_id"]), float(row["start"]), float(row["end"]),
                    score, str(row["text"])
                ))
        return sorted(results, key=lambda hit: (
            -hit.score, hit.video_id, hit.start
        ))[:limit]
