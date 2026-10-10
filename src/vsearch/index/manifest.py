"""SQLite is the manifest source of truth for video ingest and result cleanup."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS videos (
  id TEXT PRIMARY KEY,
  original_name TEXT NOT NULL,
  source_path TEXT,
  sha256 TEXT NOT NULL UNIQUE,
  duration REAL NOT NULL CHECK(duration >= 0),
  width INTEGER NOT NULL DEFAULT 0,
  height INTEGER NOT NULL DEFAULT 0,
  has_audio INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL CHECK(status IN ('pending','processing','done','failed')),
  error TEXT,
  ingest_version TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS chunks (
  video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  chunk_number INTEGER NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('processing','done','failed')),
  error TEXT,
  PRIMARY KEY (video_id, chunk_number)
);
CREATE TABLE IF NOT EXISTS frames (
  id TEXT PRIMARY KEY,
  video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  t REAL NOT NULL CHECK(t >= 0),
  scene_id INTEGER NOT NULL,
  frame_hash TEXT NOT NULL,
  thumbnail TEXT NOT NULL,
  chunk_number INTEGER NOT NULL,
  UNIQUE(video_id, t)
);
CREATE INDEX IF NOT EXISTS idx_frames_video_time ON frames(video_id, t);
"""


class Manifest:
    """Open one short-lived database connection per operation, no shared cursors."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "manifest.sqlite3"
        with self.connection() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_video(self, video_id: str) -> dict[str, object] | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
            return dict(row) if row else None

    def list_videos(self) -> list[dict[str, object]]:
        with self.connection() as conn:
            return [
                dict(row) for row in conn.execute("SELECT * FROM videos ORDER BY created_at DESC")
            ]

    def frames(self, video_id: str) -> list[dict[str, object]]:
        with self.connection() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM frames WHERE video_id=? ORDER BY t", (video_id,)
                )
            ]

    def chunks_done(self, video_id: str) -> set[int]:
        with self.connection() as conn:
            return {
                int(row[0])
                for row in conn.execute(
                    "SELECT chunk_number FROM chunks WHERE video_id=? AND status='done'",
                    (video_id,),
                )
            }

    def upsert_video(
        self,
        video_id: str,
        name: str,
        source_path: str | None,
        sha256: str,
        duration: float,
        width: int,
        height: int,
        has_audio: bool,
        version: str,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """INSERT INTO videos
                   (id,original_name,source_path,sha256,duration,width,height,
                    has_audio,status,ingest_version)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                   status=CASE WHEN ingest_version=excluded.ingest_version
                               THEN status ELSE 'pending' END,
                   ingest_version=excluded.ingest_version,
                   source_path=excluded.source_path""",
                (
                    video_id,
                    name,
                    source_path,
                    sha256,
                    duration,
                    width,
                    height,
                    int(has_audio),
                    "pending",
                    version,
                ),
            )

    def set_status(self, video_id: str, status: str, error: str | None = None) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE videos SET status=?,error=? WHERE id=?",
                (status, error[:300] if error else None, video_id),
            )

    def begin_chunk(self, video_id: str, chunk: int) -> None:
        with self.connection() as conn:
            conn.execute(
                """INSERT INTO chunks(video_id,chunk_number,status,error)
                   VALUES (?,?,'processing',NULL)
                   ON CONFLICT(video_id,chunk_number) DO UPDATE
                   SET status='processing',error=NULL""",
                (video_id, chunk),
            )

    def finish_chunk(
        self, video_id: str, chunk: int, frames: list[tuple[str, float, int, str, str]]
    ) -> None:
        """Commit frames and checkpoint atomically; partial chunks can be retried."""
        with self.connection() as conn:
            conn.execute(
                "DELETE FROM frames WHERE video_id=? AND chunk_number=?", (video_id, chunk)
            )
            conn.executemany(
                """INSERT INTO frames
                   (id,video_id,t,scene_id,frame_hash,thumbnail,chunk_number)
                   VALUES (?,?,?,?,?,?,?)""",
                [
                    (fid, video_id, t, scene, hash_value, thumb, chunk)
                    for fid, t, scene, hash_value, thumb in frames
                ],
            )
            conn.execute(
                "UPDATE chunks SET status='done',error=NULL WHERE video_id=? AND chunk_number=?",
                (video_id, chunk),
            )

    def failed_chunk(self, video_id: str, chunk: int, reason: str) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE chunks SET status='failed',error=? WHERE video_id=? AND chunk_number=?",
                (reason[:300], video_id, chunk),
            )

    def erase(self, video_id: str) -> bool:
        with self.connection() as conn:
            cursor = conn.execute("DELETE FROM videos WHERE id=?", (video_id,))
            return cursor.rowcount > 0

    def verify(self) -> list[str]:
        """Check metadata against on-disk frames; future vector stores add checks."""
        errors: list[str] = []
        with self.connection() as conn:
            rows = conn.execute("SELECT id,thumbnail FROM frames").fetchall()
        thumbs_root = (self.root / "thumbs").resolve()
        for row in rows:
            target = (self.root / str(row["thumbnail"])).resolve()
            if not target.is_relative_to(thumbs_root) or not target.is_file():
                errors.append(f"missing-or-unsafe-thumbnail:{row['id']}")
        return errors

    def repair(self) -> list[str]:
        """Mark affected videos failed rather than silently recreating metadata."""
        errors = self.verify()
        affected = {item.rsplit(":", 1)[-1] for item in errors}
        with self.connection() as conn:
            for fid in affected:
                row = conn.execute("SELECT video_id FROM frames WHERE id=?", (fid,)).fetchone()
                if row:
                    conn.execute(
                        "UPDATE videos SET status='failed',error='missing thumbnails' WHERE id=?",
                        (row["video_id"],),
                    )
        return errors
