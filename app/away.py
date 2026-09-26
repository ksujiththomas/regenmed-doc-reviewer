"""Away mode: watch a Google Drive folder on a schedule.

Flow: poll Drive "watch" folder for PDFs -> review each with the normal
pipeline -> upload a CSV report to Reports/ -> move the PDF to Processed/
(clean) or Needs attention/ (errors/warnings) -> record in review log.

All sqlite access happens in the scheduler thread (single worker), so no
concurrency concerns.
"""
import csv
import io
import time
from datetime import datetime

from . import drive_client, history
from .pipeline import run

DEFAULT_INTERVAL_MIN = 30
ROOT_NAME = "RegenMed Reviewer"


# ---------------------------------------------------------------- config

def get_config():
    cfg = {"enabled": "0", "interval_min": str(DEFAULT_INTERVAL_MIN)}
    c = history._conn()
    try:
        for key, value in c.execute("SELECT key, value FROM settings"):
            cfg[key] = value
    finally:
        c.close()
    return cfg


def save_config(**kwargs):
    c = history._conn()
    try:
        for key, value in kwargs.items():
            c.execute("INSERT OR REPLACE INTO settings(key, value) VALUES(?,?)",
                      (key, str(value)))
        c.commit()
    finally:
        c.close()


def clear_token():
    c = history._conn()
    try:
        c.execute("DELETE FROM settings WHERE key IN "
                  "('refresh_token','account_email')")
        c.commit()
    finally:
        c.close()


# ---------------------------------------------------------------- runs

def _ensure_runs_table():
    c = history._conn()
    try:
        c.execute("""CREATE TABLE IF NOT EXISTS away_runs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at REAL, finished_at REAL,
            checked INTEGER, processed INTEGER,
            clean INTEGER, attention INTEGER, failed INTEGER,
            status TEXT, note TEXT)""")
        c.commit()
    finally:
        c.close()


def record_run(started, checked, processed, clean, attention, failed,
               status, note=""):
    _ensure_runs_table()
    c = history._conn()
    try:
        c.execute("""INSERT INTO away_runs(started_at, finished_at, checked,
                     processed, clean, attention, failed, status, note)
                     VALUES(?,?,?,?,?,?,?,?,?)""",
                  (started, time.time(), checked, processed, clean,
                   attention, failed, status, note))
        c.commit()
    finally:
        c.close()


def recent_runs(limit=10):
    _ensure_runs_table()
    c = history._conn()
    try:
        rows = c.execute("""SELECT started_at, finished_at, checked,
                            processed, clean, attention, failed, status, note
                            FROM away_runs ORDER BY id DESC LIMIT ?""",
                         (limit,)).fetchall()
    finally:
        c.close()
    out = []
    for r in rows:
        out.append({
            "when": datetime.fromtimestamp(r[0]).strftime("%Y-%m-%d %H:%M"),
            "secs": round((r[1] or r[0]) - r[0], 1),
            "checked": r[2], "processed": r[3], "clean": r[4],
            "attention": r[5], "failed": r[6],
            "status": r[7], "note": r[8] or "",
        })
    return out


# ---------------------------------------------------------------- folders

def setup_folders(svc, parent_id, parent_name):
    """Create Processed / Needs attention / Reports inside parent_id."""
    processed = drive_client.ensure_folder(svc, "Processed", parent_id)
    attention = drive_client.ensure_folder(svc, "Needs attention", parent_id)
    reports = drive_client.ensure_folder(svc, "Reports", parent_id)
    save_config(watch_id=parent_id, watch_name=parent_name,
                processed_id=processed, attention_id=attention,
                reports_id=reports)
    return {"processed": processed, "attention": attention,
            "reports": reports}


def setup_default_tree(svc):
    """Create 'RegenMed Reviewer' in Drive root with the four subfolders."""
    root = drive_client.ensure_folder(svc, ROOT_NAME, "root")
    watch = drive_client.ensure_folder(svc, "Watch", root)
    setup_folders(svc, watch, ROOT_NAME + " / Watch")
    save_config(root_id=root)
    return ROOT_NAME


# ---------------------------------------------------------------- review

def _report_csv(filename, form_code, form_name, issues):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["filename", "form", "status", "severity", "field", "message"])
    status = "PASS" if not issues else (
        "ERROR" if any(i["severity"] == "error" for i in issues)
        else "WARNING")
    if not issues:
        w.writerow([filename, form_name, status, "", "", ""])
    for i in issues:
        w.writerow([filename, form_name, status,
                    i["severity"], i["field"], i["message"]])
    return buf.getvalue().encode("utf-8")


def _process_one(svc, meta, cfg):
    """Review one Drive file; returns (outcome, note)."""
    file_id = meta["id"]
    name = meta["name"]
    parents = meta.get("parents", [])
    from_parent = parents[0] if parents else cfg["watch_id"]

    pdf_bytes = drive_client.download_pdf(svc, file_id)
    if len(pdf_bytes) > 15 * 1024 * 1024:
        return "skipped", "over 15 MB"

    try:
        r = run(pdf_bytes, annotate_pages=False)
    except Exception as exc:  # noqa: BLE001 - keep the file, just note it
        return "failed", str(exc)[:120]

    if r["form_code"] is None:
        # not one of our forms: leave it in Watch, note in run log
        return "failed", r.get("note") or "form not recognized"

    issues = r["issues"]
    errors = [i for i in issues if i["severity"] == "error"]
    clean = not issues

    report_name = name.rsplit(".", 1)[0] + "_report.csv"
    csv_bytes = _report_csv(name, r["form_code"], r["form_name"], issues)
    drive_client.upload_report(svc, report_name, csv_bytes,
                               cfg["reports_id"])

    dest = cfg["processed_id"] if clean else cfg["attention_id"]
    drive_client.move_file(svc, file_id, from_parent, dest)

    history.record(history.sha256_hex(pdf_bytes), name,
                   r["form_code"], r["form_name"],
                   len(errors), len(issues) - len(errors))
    return ("clean" if clean else "attention",
            f"{len(errors)} errors, {len(issues) - len(errors)} warnings")


# ---------------------------------------------------------------- poll

def poll_drive():
    """The scheduled job. Safe to call directly (Run now button)."""
    started = time.time()
    cfg = get_config()
    if cfg.get("enabled") != "1":
        return
    token = cfg.get("refresh_token")
    if not token:
        record_run(started, 0, 0, 0, 0, 0, "error", "Drive not connected")
        return
    if not all(cfg.get(k) for k in
               ("watch_id", "processed_id", "attention_id", "reports_id")):
        record_run(started, 0, 0, 0, 0, 0, "error",
                   "Folders not set up")
        return
    try:
        svc = drive_client.drive_service(token)
        files = drive_client.list_pdfs(svc, cfg["watch_id"])
    except Exception as exc:  # noqa: BLE001 - auth or API failure
        msg = str(exc)
        if "invalid_grant" in msg or "RefreshError" in type(exc).__name__:
            clear_token()
            record_run(started, 0, 0, 0, 0, 0, "error",
                       "Drive authorization expired — reconnect")
        else:
            record_run(started, 0, 0, 0, 0, 0, "error",
                       f"Drive error: {msg[:120]}")
        return

    checked, clean, attention, failed = len(files), 0, 0, 0
    notes = []
    for meta in files:
        try:
            outcome, note = _process_one(svc, meta, cfg)
        except Exception as exc:  # noqa: BLE001 - never kill the run
            outcome, note = "failed", str(exc)[:120]
        if outcome == "clean":
            clean += 1
        elif outcome == "attention":
            attention += 1
        else:
            failed += 1
            notes.append(f"{meta['name']}: {note}")

    record_run(started, checked, clean + attention, clean, attention,
               failed, "ok" if not failed else "partial",
               "; ".join(notes[:3]))
    save_config(last_run=datetime.now().strftime("%Y-%m-%d %H:%M"))
