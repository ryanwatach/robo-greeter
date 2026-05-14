import csv
import os
import sqlite3
from typing import List, Optional, Tuple

import numpy as np

from utils.logger import setup_logger

log = setup_logger("robo-greeter")

SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    last_seen TEXT DEFAULT (datetime('now')),
    visit_count INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    captured_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (person_id) REFERENCES persons(id)
);

CREATE TABLE IF NOT EXISTS check_ins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    checked_in_at TEXT DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (person_id) REFERENCES persons(id)
);
"""


class FaceDatabase:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        log.info("Database initialized at %s", db_path)

    def add_person(self, name: str, embeddings: List[np.ndarray]) -> int:
        cur = self.conn.execute(
            "INSERT INTO persons (name) VALUES (?)", (name,)
        )
        person_id = cur.lastrowid
        for emb in embeddings:
            self.conn.execute(
                "INSERT INTO embeddings (person_id, embedding) VALUES (?, ?)",
                (person_id, self._serialize(emb)),
            )
        self.conn.commit()
        log.info("Added person '%s' (id=%d) with %d embeddings", name, person_id, len(embeddings))
        return person_id

    def get_all_persons(self) -> List[Tuple[int, str, np.ndarray]]:
        rows = self.conn.execute("SELECT id, name FROM persons").fetchall()
        result = []
        for pid, name in rows:
            emb_rows = self.conn.execute(
                "SELECT embedding FROM embeddings WHERE person_id = ?", (pid,)
            ).fetchall()
            if not emb_rows:
                continue
            embeddings = [self._deserialize(r[0]) for r in emb_rows]
            avg = np.mean(embeddings, axis=0)
            result.append((pid, name, avg))
        return result

    def update_last_seen(self, person_id: int):
        self.conn.execute(
            "UPDATE persons SET last_seen = datetime('now'), "
            "visit_count = visit_count + 1 WHERE id = ?",
            (person_id,),
        )
        self.conn.commit()

    def add_embedding(self, person_id: int, embedding: np.ndarray):
        self.conn.execute(
            "INSERT INTO embeddings (person_id, embedding) VALUES (?, ?)",
            (person_id, self._serialize(embedding)),
        )
        self.conn.commit()

    def get_person_by_id(self, person_id: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT id, name, created_at, last_seen, visit_count FROM persons WHERE id = ?",
            (person_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "name": row[1],
            "created_at": row[2],
            "last_seen": row[3],
            "visit_count": row[4],
        }

    def person_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]

    # ── Check-ins ────────────────────────────────────────────────

    def record_check_in(self, person_id: int, name: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO check_ins (person_id, name) VALUES (?, ?)",
            (person_id, name),
        )
        self.conn.commit()
        log.info("Recorded check-in: %s (person_id=%d)", name, person_id)
        return cur.lastrowid

    def get_check_ins(self, limit: int = 500) -> List[Tuple[int, int, str, str]]:
        """Return rows of (id, person_id, name, checked_in_at), newest first."""
        return self.conn.execute(
            "SELECT id, person_id, name, checked_in_at "
            "FROM check_ins ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def get_check_ins_for_person(self, person_id: int, limit: int = 50) -> List[Tuple[int, str]]:
        """Return rows of (id, checked_in_at) for one person, newest first."""
        return self.conn.execute(
            "SELECT id, checked_in_at FROM check_ins "
            "WHERE person_id = ? ORDER BY id DESC LIMIT ?",
            (person_id, limit),
        ).fetchall()

    def export_check_ins_csv(self, path: str) -> int:
        """Dump every check-in to a CSV file. Returns row count."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        rows = self.conn.execute(
            "SELECT id, person_id, name, checked_in_at "
            "FROM check_ins ORDER BY id DESC"
        ).fetchall()
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "person_id", "name", "checked_in_at"])
            writer.writerows(rows)
        return len(rows)

    @staticmethod
    def _serialize(emb: np.ndarray) -> bytes:
        return emb.astype(np.float64).tobytes()

    @staticmethod
    def _deserialize(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float64)

    def close(self):
        self.conn.close()
