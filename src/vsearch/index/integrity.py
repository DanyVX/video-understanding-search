"""Cross-store integrity verification and conservative orphan-vector repair."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from vsearch.index.manifest import Manifest
from vsearch.index.vectors import SQLiteVectorStore


def verify_all(root: Path) -> list[str]:
    """Check thumbnails, foreign keys, vector references and stored dimensions."""
    try:
        manifest = Manifest(root)
        problems = manifest.verify()
        with manifest.connection() as conn:
            foreign = conn.execute("PRAGMA foreign_key_check").fetchall()
            if foreign:
                problems.append(f"manifest-foreign-key-errors:{len(foreign)}")
            frames = {
                str(row["id"])
                for row in conn.execute("SELECT id FROM frames").fetchall()
            }
            video_ids = {
                str(row["id"])
                for row in conn.execute("SELECT id FROM videos").fetchall()
            }
        vectors = SQLiteVectorStore(root)
        problems.extend(vectors.verify(video_ids))
        with vectors._connect() as conn:
            rows = conn.execute(
                """SELECT v.collection,v.item_id,v.video_id,v.embedding,
                          c.dimensions
                   FROM vectors v JOIN collections c ON v.collection=c.name"""
            ).fetchall()
        for collection, item_id, video_id, blob, dimensions in rows:
            if collection == "visual" and str(item_id) not in frames:
                problems.append(f"orphan-frame-vector:{video_id}:{item_id}")
            if len(blob) != int(dimensions) * 4:
                problems.append(f"bad-vector-dimensions:{video_id}:{item_id}")
        return sorted(set(problems))
    except sqlite3.DatabaseError:
        return ["database-corruption-requires-restore"]


def prune_orphan_vectors(root: Path) -> int:
    """Explicitly remove vectors for missing videos or deleted frame IDs."""
    manifest = Manifest(root)
    with manifest.connection() as conn:
        videos = {str(row[0]) for row in conn.execute("SELECT id FROM videos")}
        frames = {str(row[0]) for row in conn.execute("SELECT id FROM frames")}
    store = SQLiteVectorStore(root)
    with store.lock, store._connect() as conn:
        rows = conn.execute(
            "SELECT collection,item_id,video_id FROM vectors"
        ).fetchall()
        bad = [
            (str(collection), str(item_id))
            for collection, item_id, video_id in rows
            if str(video_id) not in videos
            or (collection == "visual" and str(item_id) not in frames)
        ]
        conn.executemany(
            "DELETE FROM vectors WHERE collection=? AND item_id=?", bad
        )
    return len(bad)
