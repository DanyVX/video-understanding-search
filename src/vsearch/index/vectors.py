"""Model-isolated float32 vector storage with optional FAISS search acceleration."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

import numpy as np

DDL = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS collections(
  name TEXT PRIMARY KEY,
  model_key TEXT NOT NULL,
  dimensions INTEGER NOT NULL CHECK(dimensions > 0)
);
CREATE TABLE IF NOT EXISTS vectors(
  collection TEXT NOT NULL REFERENCES collections(name) ON DELETE CASCADE,
  item_id TEXT NOT NULL,
  video_id TEXT NOT NULL,
  t REAL NOT NULL CHECK(t>=0),
  embedding BLOB NOT NULL,
  payload TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY(collection,item_id)
);
CREATE INDEX IF NOT EXISTS idx_vectors_video ON vectors(video_id);
"""


class IndexMismatch(ValueError):
    """Prevent silently mixing incompatible model spaces or dimensions."""


@dataclass(frozen=True)
class VectorHit:
    item_id: str
    video_id: str
    t: float
    score: float
    payload: dict[str, Any]


class VectorStore(Protocol):
    def add(
        self,
        collection: str,
        model_key: str,
        vectors: list[tuple[str, str, float, np.ndarray, dict[str, Any]]],
    ) -> None: ...

    def search(
        self, collection: str, model_key: str, query: np.ndarray, limit: int = 10
    ) -> list[VectorHit]: ...

    def delete_video(self, video_id: str) -> int: ...


def normalize(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Embedding must be a nonempty finite 1-D array")
    norm = float(np.linalg.norm(values))
    if norm <= 1e-12:
        raise ValueError("Zero embedding is not searchable")
    return values / norm


class SQLiteVectorStore:
    """Persistent, exact cosine retrieval. FAISS is an optional accelerator."""

    def __init__(self, root: Path, *, use_faiss: bool = False) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "vectors.sqlite3"
        self.lock = RLock()
        self.use_faiss = use_faiss
        with self._connect() as conn:
            conn.executescript(DDL)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _assert_collection(
        self, conn: sqlite3.Connection, name: str, model_key: str, dims: int
    ) -> None:
        existing = conn.execute(
            "SELECT model_key,dimensions FROM collections WHERE name=?", (name,)
        ).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO collections(name,model_key,dimensions) VALUES (?,?,?)",
                (name, model_key, dims),
            )
        elif existing != (model_key, dims):
            raise IndexMismatch("Incompatible model or dimension; use a new collection or reindex")

    def add(
        self,
        collection: str,
        model_key: str,
        vectors: list[tuple[str, str, float, np.ndarray, dict[str, Any]]],
    ) -> None:
        if not vectors:
            return
        if not collection or not model_key:
            raise ValueError("Collection and model signature must be nonempty")
        with self.lock, self._connect() as conn:
            prepared = []
            for item_id, video_id, t, vector, payload in vectors:
                emb = normalize(vector)
                if t < 0 or not np.isfinite(t):
                    raise ValueError("Invalid timestamp")
                self._assert_collection(conn, collection, model_key, int(emb.size))
                prepared.append(
                    (
                        collection,
                        item_id,
                        video_id,
                        t,
                        emb.tobytes(),
                        json.dumps(payload, ensure_ascii=False),
                    )
                )
            conn.executemany(
                """INSERT INTO vectors(collection,item_id,video_id,t,embedding,payload)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(collection,item_id) DO UPDATE SET
                   video_id=excluded.video_id,t=excluded.t,embedding=excluded.embedding,
                   payload=excluded.payload""",
                prepared,
            )

    def search(
        self, collection: str, model_key: str, query: np.ndarray, limit: int = 10
    ) -> list[VectorHit]:
        if limit <= 0:
            return []
        vector = normalize(query)
        with self.lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT model_key,dimensions FROM collections WHERE name=?", (collection,)
            ).fetchone()
            if existing is None:
                return []
            if existing != (model_key, int(vector.size)):
                raise IndexMismatch("Query model/dimension incompatible with collection")
            rows = conn.execute(
                """SELECT item_id,video_id,t,embedding,payload FROM vectors
                   WHERE collection=? ORDER BY item_id""",
                (collection,),
            ).fetchall()
        if not rows:
            return []
        matrix = np.vstack([np.frombuffer(row[3], dtype=np.float32) for row in rows])
        if self.use_faiss:
            try:
                import faiss
            except ImportError as exc:
                raise RuntimeError("Install vsearch[vector] for FAISS acceleration") from exc
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(np.ascontiguousarray(matrix))
            scores, indices = index.search(vector.reshape(1, -1), min(limit, len(rows)))
            order = list(zip(indices[0].tolist(), scores[0].tolist(), strict=False))
        else:
            scores_array = matrix @ vector
            rank = sorted(range(len(rows)), key=lambda i: (-float(scores_array[i]), rows[i][0]))
            order = [(i, float(scores_array[i])) for i in rank[:limit]]
        return [
            VectorHit(
                str(rows[i][0]),
                str(rows[i][1]),
                float(rows[i][2]),
                float(score),
                json.loads(rows[i][4]),
            )
            for i, score in order
            if i >= 0
        ]

    def delete_video(self, video_id: str) -> int:
        with self.lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM vectors WHERE video_id=?", (video_id,))
            return cursor.rowcount

    def count(self, collection: str) -> int:
        with self.lock, self._connect() as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM vectors WHERE collection=?", (collection,)
            ).fetchone()
            return int(result[0]) if result else 0

    def verify(self, video_ids: set[str]) -> list[str]:
        with self.lock, self._connect() as conn:
            entries = conn.execute("SELECT DISTINCT video_id FROM vectors").fetchall()
        return sorted(f"orphan-vector:{row[0]}" for row in entries if row[0] not in video_ids)
