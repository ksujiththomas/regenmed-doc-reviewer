#!/usr/bin/env python3
"""Drive Away-mode watcher (run hourly by cron).

Checks the Watch folder for new PDFs, reviews them through the Render
app's /api/review endpoint, uploads a CSV report per file to Reports/,
and moves each PDF to Processed/ (clean) or Needs attention/ (issues).
Unrecognized files are left in place and noted.

State (processed file ids) lives in state.json next to this script.
"""
import csv
import io
import json
import os
import subprocess
import sys
import urllib.request

APP_URL = "https://regenmed-doc-reviewer.onrender.com"

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(HERE, "state.json")
TMP = "/tmp/away_watch"
MAX_PER_RUN = 10

# Resolved per run from the app's /api/away/config:
WATCH_ID = None
PROCESSED_ID = None
ATTENTION_ID = None
REPORTS_ID = None


def gws(*args):
    """Run hatch_gws_cli drive <args>, return parsed JSON."""
    out = subprocess.run(["hatch_gws_cli", "drive"] + list(args),
                         capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"drive CLI failed: {out.stderr[:200]}")
    return json.loads(out.stdout)


def get_config():
    """Fetch Away-mode config from the app. Raises on failure."""
    req = urllib.request.Request(APP_URL + "/api/away/config")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def ensure_subfolders(watch_id):
    """Find or create Processed / Needs attention / Reports under watch."""
    global PROCESSED_ID, ATTENTION_ID, REPORTS_ID
    resp = gws("files", "list", "--params", json.dumps({
        "q": f"'{watch_id}' in parents and trashed=false "
             f"and mimeType='application/vnd.google-apps.folder'",
        "pageSize": 50,
        "fields": "files(id,name)",
    }))
    found = {f["name"]: f["id"] for f in resp.get("files", [])}
    ids = {}
    for name in ("Processed", "Needs attention", "Reports"):
        if name in found:
            ids[name] = found[name]
        else:
            created = gws("files", "create", "--params",
                          '{"ignoreDefaultVisibility":true}', "--json",
                          json.dumps({
                              "name": name,
                              "mimeType": "application/vnd.google-apps.folder",
                              "parents": [watch_id]}))
            ids[name] = created["id"]
            print(f"Created '{name}' subfolder.")
    PROCESSED_ID = ids["Processed"]
    ATTENTION_ID = ids["Needs attention"]
    REPORTS_ID = ids["Reports"]


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"processed_ids": []}


def save_state(state):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, STATE_PATH)


def list_new_pdfs(state):
    done = set(state["processed_ids"])
    resp = gws("files", "list", "--params", json.dumps({
        "q": f"'{WATCH_ID}' in parents and trashed=false "
             f"and mimeType='application/pdf'",
        "pageSize": 50,
        "fields": "files(id,name,parents,modifiedTime,size)",
        "orderBy": "modifiedTime",
    }))
    return [f for f in resp.get("files", []) if f["id"] not in done]


def download_pdf(file_id, name):
    os.makedirs(TMP, exist_ok=True)
    # sanitize filename for local disk
    safe = "".join(c for c in name if c.isalnum() or c in "._- ")[:80]
    path = os.path.join(TMP, f"{file_id}_{safe}")
    gws("files", "get", "--params",
        json.dumps({"fileId": file_id, "alt": "media"}),
        "--output", path)
    with open(path, "rb") as f:
        return path, f.read()


def review_batch(items):
    """items: list of (file_id, name, bytes). Returns {file_id: result}."""
    # Write files to disk and POST with curl (robust multipart).
    paths = []
    for fid, name, data in items:
        p = os.path.join(TMP, f"up_{fid}.pdf")
        with open(p, "wb") as f:
            f.write(data)
        paths.append(p)
    out_path = os.path.join(TMP, "api_result.json")
    cmd = ["curl", "-s", "--max-time", "300", "-X", "POST",
           APP_URL + "/api/review", "-o", out_path, "-w", "%{http_code}"]
    for p in paths:
        cmd += ["-F", f"pdf=@{p}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=330)
    code = (proc.stdout or "").strip()
    if code != "200":
        raise RuntimeError(f"review API HTTP {code}: "
                           f"{(proc.stderr or '')[:150]}")
    with open(out_path) as f:
        data = json.load(f)
    out = {}
    for (fid, name, _), res in zip(items, data.get("results", [])):
        out[fid] = res
    return out


def report_csv(name, result):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["filename", "form", "status", "severity", "field", "message"])
    if not result.get("ok"):
        w.writerow([name, "", "ERROR", "", "", result.get("error", "")])
    else:
        status = "PASS" if result["passed"] else "FAIL"
        issues = result["errors"] + result["warnings"]
        if not issues:
            w.writerow([name, result["form_name"], status, "", "", ""])
        for iss in issues:
            w.writerow([name, result["form_name"], status,
                        iss["severity"], iss["field"], iss["message"]])
    return buf.getvalue().encode("utf-8")


def upload_report(name, content):
    report_name = name.rsplit(".", 1)[0] + "_report.csv"
    os.makedirs(TMP, exist_ok=True)
    path = os.path.join(TMP, report_name)
    with open(path, "wb") as f:
        f.write(content)
    res = gws("+upload", path, "--parent", REPORTS_ID, "--name", report_name)
    return res


def move_file(file_id, from_parent, to_parent):
    gws("files", "update", "--params", json.dumps({
        "fileId": file_id, "addParents": to_parent,
        "removeParents": from_parent}),
        "--json", "{}")


def main():
    global WATCH_ID
    try:
        cfg = get_config()
    except Exception as exc:
        print(f"Could not read Away-mode config from app: {exc}")
        return 1
    if not cfg.get("enabled"):
        print("Away mode is disabled — skipping.")
        return 0
    if not cfg.get("folder_id"):
        print("No Drive folder attached — skipping.")
        return 0
    WATCH_ID = cfg["folder_id"]
    try:
        ensure_subfolders(WATCH_ID)
    except Exception as exc:
        print(f"Could not prepare Drive subfolders: {exc}")
        return 1

    state = load_state()
    new_files = list_new_pdfs(state)[:MAX_PER_RUN]
    if not new_files:
        print("No new PDFs in the Watch folder.")
        return 0

    print(f"Found {len(new_files)} new PDF(s). Downloading…")
    items = []
    for meta in new_files:
        size = int(meta.get("size", 0) or 0)
        if size > 15 * 1024 * 1024:
            print(f"  skip {meta['name']}: over 15 MB")
            state["processed_ids"].append(meta["id"])
            continue
        try:
            _, data = download_pdf(meta["id"], meta["name"])
        except Exception as exc:
            print(f"  download failed {meta['name']}: {exc}")
            continue
        items.append((meta["id"], meta["name"], data))

    if not items:
        save_state(state)
        return 0

    print("Reviewing via app…")
    results = review_batch(items)

    clean_n = att_n = fail_n = 0
    for fid, name, _ in items:
        res = results.get(fid, {"ok": False, "error": "no result"})
        parents = next((m.get("parents", []) for m in new_files
                        if m["id"] == fid), [])
        from_parent = parents[0] if parents else WATCH_ID
        if not res.get("ok"):
            fail_n += 1
            print(f"  {name}: FAILED ({res.get('error','')[:80]}) — left in place")
        else:
            csv_bytes = report_csv(name, res)
            try:
                upload_report(name, csv_bytes)
            except Exception as exc:
                print(f"  {name}: report upload failed: {exc}")
            dest = PROCESSED_ID if res["passed"] else ATTENTION_ID
            try:
                move_file(fid, from_parent, dest)
            except Exception as exc:
                print(f"  {name}: move failed: {exc}")
                continue
            if res["passed"]:
                clean_n += 1
            else:
                att_n += 1
            print(f"  {name}: {'CLEAN' if res['passed'] else 'ISSUES'} "
                  f"({res['n_errors']} errors, {res['n_warnings']} warnings)")
        state["processed_ids"].append(fid)

    save_state(state)
    print(f"Done: {clean_n} clean, {att_n} need attention, "
          f"{fail_n} failed/left in place.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
