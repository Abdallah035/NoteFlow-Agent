"""Database layer: connection setup and schema."""

import json
import sqlite3
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT NOT NULL,
    body       TEXT NOT NULL,
    tags       TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    action    TEXT NOT NULL,
    note_id   INTEGER,
    before    TEXT,
    after     TEXT,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id    INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    embedding  BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_note ON chunks(note_id);
"""


def connect(db_path: str = "noteflow.db") -> sqlite3.Connection:
    """Open a SQLite connection with the settings NoteFlow depends on.

    Args:
        db_path: Path to the database file. Use ":memory:" for a throwaway
            in-memory database (this is what the tests do).

    Returns:
        An open connection. The caller is responsible for closing it.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create the tables if they don't exist. Safe to call on every startup.

    Args:
        conn: An open connection from connect().
    """
    conn.executescript(SCHEMA)
    conn.commit()


def _utc_now() -> str:
    """Return the current UTC time as text, like '2026-08-01T14:30:00'."""
    now = datetime.now(timezone.utc)
    return now.replace(microsecond=0, tzinfo=None).isoformat()


def _audit(conn, action, note_id, before, after, ts):
    """Save one row in audit_log. The caller does the commit."""
    if before is not None:
        before = json.dumps(before, ensure_ascii=False)
    if after is not None:
        after = json.dumps(after, ensure_ascii=False)

    conn.execute(
        "INSERT INTO audit_log (action, note_id, before, after, timestamp) "
        "VALUES (?, ?, ?, ?, ?)",
        (action, note_id, before, after, ts),
    )


def add_note(conn, title: str, body: str, tags: list[str] | None = None) -> int:
    """Add a new note and return its id."""
    if tags is None:
        tags = []

    now = _utc_now()
    tags_json = json.dumps(tags, ensure_ascii=False)

    cur = conn.execute(
        "INSERT INTO notes (title, body, tags, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (title, body, tags_json, now, now),
    )
    note_id = cur.lastrowid

    after = {"id": note_id, "title": title, "body": body, "tags": tags}
    _audit(conn, "add", note_id, None, after, now)

    conn.commit()
    return note_id


def _row_to_note(row) -> dict:
    """Turn a database row into a normal dict with tags as a real list."""
    return {
        "id": row["id"],
        "title": row["title"],
        "body": row["body"],
        "tags": json.loads(row["tags"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def get_note(conn, note_id: int) -> dict | None:
    """Return one note as a dict, or None if there is no note with that id."""
    row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
    if row is None:
        return None
    return _row_to_note(row)


def update_note(conn, note_id, title=None, body=None, tags=None,
                body_mode="replace", tags_mode="replace") -> dict | None:
    """Update a note. Only the fields you pass are changed.

    body_mode:  "replace" swaps the body, "append" adds to the end.
    tags_mode:  "replace" swaps the tags, "append" adds new ones.
    Returns the updated note, or None if the note does not exist.
    """
    before = get_note(conn, note_id)
    if before is None:
        return None

    new_title = title if title is not None else before["title"]

    if tags is None:
        new_tags = before["tags"]
    elif tags_mode == "append":
        new_tags = before["tags"] + [t for t in tags if t not in before["tags"]]
    else:
        new_tags = tags

    if body is None:
        new_body = before["body"]
    elif body_mode == "append":
        new_body = before["body"] + "\n" + body
    else:
        new_body = body

    now = _utc_now()
    conn.execute(
        "UPDATE notes SET title=?, body=?, tags=?, updated_at=? WHERE id=?",
        (new_title, new_body, json.dumps(new_tags, ensure_ascii=False), now, note_id),
    )

    after = get_note(conn, note_id)
    _audit(conn, "update", note_id, before, after, now)
    conn.commit()
    return after


def search_notes(conn, query=None, tags=None, date_from=None, date_to=None,
                 limit=50) -> list[dict]:
    """Find notes by keyword, tags, or date. No filters means list all.

    Dates are 'YYYY-MM-DD' strings. Results are newest first.
    """
    sql = "SELECT * FROM notes WHERE 1=1"
    params = []

    if query:
        sql += " AND (title LIKE ? OR body LIKE ?)"
        params.append(f"%{query}%")
        params.append(f"%{query}%")

    if tags:
        for tag in tags:
            # Tags are stored as '["work","urgent"]', so we search for the
            # tag WITH its quotes. Without them 'work' would match 'homework'.
            sql += " AND tags LIKE ?"
            params.append(f'%"{tag}"%')

    if date_from:
        sql += " AND created_at >= ?"
        params.append(date_from)

    if date_to:
        # created_at has a time part, so add the end of the day to include it.
        sql += " AND created_at <= ?"
        params.append(date_to + "T23:59:59")

    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    return [_row_to_note(row) for row in rows]


def delete_note(conn, note_id: int) -> dict | None:
    """Delete a note. Returns the deleted note, or None if it did not exist."""
    before = get_note(conn, note_id)
    if before is None:
        return None

    conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    _audit(conn, "delete", note_id, before, None, _utc_now())
    conn.commit()
    return before
