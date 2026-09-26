"""Review log: SHA-256 based history of reviewed files.

Duplicate detection: if the same file bytes were reviewed before, the
uploader is told when and what the result was, instead of silently
re-processing.

The DB lives in DATA_DIR (a Render persistent disk mount, e.g. /app/data)
so the log and Away-mode tokens survive deploys. Falls back to the app
directory when DATA_DIR is unset or not writable.
"""
import hashlib
import os
import sqlite3
import time


def _data_dir():
    d = os.environ.get("DATA_DIR", "/app/data")
    try:
        os.makedirs(d, exist_ok=True)
        # verify writability
        probe = os.path.join(d, ".writetest")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        return d
    except OSError:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


DB_PATH = os.path.join(_data_dir(), "review_log.db")


def _conn():
    # timeout=30: batch worker threads may briefly contend on writes
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.execute("""CREATE TABLE IF NOT EXISTS reviews(
        sha256 TEXT PRIMARY KEY,
        filename TEXT, form_code TEXT, form_name TEXT,
        n_errors INTEGER, n_warnings INTEGER, reviewed_at REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS settings(
        key TEXT PRIMARY KEY, value TEXT)""")
    return c


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_enabled() -> bool:
    c = _conn()
    try:
        row = c.execute("SELECT value FROM settings WHERE key='log_enabled'").fetchone()
        return row is None or row[0] == "1"
    finally:
        c.close()


def set_enabled(on: bool):
    c = _conn()
    try:
        c.execute("INSERT OR REPLACE INTO settings(key, value) VALUES('log_enabled', ?)",
                  ("1" if on else "0",))
        c.commit()
    finally:
        c.close()


def seed_away_from_env():
    """Seed away-mode settings from env vars on a fresh DB.

    Render's ephemeral filesystem wipes sqlite on every deploy; env vars
    survive. On startup, fill any ABSENT away_* keys from AWAY_ENABLED /
    AWAY_FOLDER_ID / AWAY_FOLDER_NAME. Never overrides values set at
    runtime via the /away page or API. INSERT OR IGNORE keeps it safe
    under gunicorn's multiple workers.
    """
    import os
    raw_enabled = os.environ.get("AWAY_ENABLED", "").strip().lower()
    defaults = {
        "away_enabled": "1" if raw_enabled in ("1", "true", "yes", "on") else "",
        "away_folder_id": os.environ.get("AWAY_FOLDER_ID", "").strip(),
        "away_folder_name": os.environ.get("AWAY_FOLDER_NAME", "").strip(),
    }
    c = _conn()
    try:
        for key, val in defaults.items():
            if val:
                c.execute("INSERT OR IGNORE INTO settings(key, value) VALUES(?, ?)",
                          (key, val))
        c.commit()
    finally:
        c.close()


def get_setting(key: str, default: str = "") -> str:
    c = _conn()
    try:
        row = c.execute("SELECT value FROM settings WHERE key=?",
                        (key,)).fetchone()
        return row[0] if row else default
    finally:
        c.close()


def set_setting(key: str, value: str):
    c = _conn()
    try:
        c.execute("INSERT OR REPLACE INTO settings(key, value) VALUES(?, ?)",
                  (key, value))
        c.commit()
    finally:
        c.close()


def find(sha: str):
    """Return the previous review record for these exact file bytes, or None."""
    c = _conn()
    try:
        row = c.execute(
            "SELECT filename, form_code, form_name, n_errors, n_warnings, reviewed_at"
            " FROM reviews WHERE sha256=?", (sha,)).fetchone()
        if not row:
            return None
        return {"filename": row[0], "form_code": row[1], "form_name": row[2],
                "n_errors": row[3], "n_warnings": row[4], "reviewed_at": row[5]}
    finally:
        c.close()


def record(sha: str, filename: str, form_code, form_name,
           n_errors: int, n_warnings: int):
    c = _conn()
    try:
        c.execute(
            "INSERT OR REPLACE INTO reviews(sha256, filename, form_code, form_name,"
            " n_errors, n_warnings, reviewed_at) VALUES(?,?,?,?,?,?,?)",
            (sha, filename, form_code, form_name, n_errors, n_warnings,
             time.time()))
        c.commit()
    finally:
        c.close()


def clear():
    c = _conn()
    try:
        c.execute("DELETE FROM reviews")
        c.commit()
    finally:
        c.close()


def count() -> int:
    c = _conn()
    try:
        return c.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    finally:
        c.close()


def recent(limit: int = 50):
    c = _conn()
    try:
        rows = c.execute(
            "SELECT filename, form_code, form_name, n_errors, n_warnings, reviewed_at"
            " FROM reviews ORDER BY reviewed_at DESC LIMIT ?", (limit,)).fetchall()
        return [{"filename": r[0], "form_code": r[1], "form_name": r[2],
                 "n_errors": r[3], "n_warnings": r[4], "reviewed_at": r[5]}
                for r in rows]
    finally:
        c.close()
