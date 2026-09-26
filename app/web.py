"""Flask web app: upload a processing-form PDF, get a validation report."""
import base64
import csv
import io
import os
import uuid

import cv2
from flask import Flask, Response, redirect, render_template, request, url_for

from . import history
from .pipeline import run, ReviewError

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 160 * 1024 * 1024  # 10 files x 15 MB + overhead

# In-memory cache of batch results for CSV download (ephemeral, fine).
BATCH_CACHE = {}


@app.errorhandler(413)
def too_large(_):
    return render_template("index.html",
                           error="Total upload is too large — please keep it under "
                                 "150 MB (max 10 files, 15 MB each)."), 413


@app.context_processor
def inject_log_state():
    return {"log_enabled": history.is_enabled(),
            "log_count": history.count()}


def _img_to_data_uri(bgr_img):
    ok, buf = cv2.imencode(".png", bgr_img)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


def _dup_check(pdf_bytes, filename):
    """Return prior review record if these exact bytes were seen before."""
    if not history.is_enabled():
        return None
    rec = history.find(history.sha256_hex(pdf_bytes))
    if rec:
        from datetime import datetime
        rec["when"] = datetime.fromtimestamp(rec["reviewed_at"]).strftime(
            "%Y-%m-%d %H:%M")
    return rec


def _log_review(pdf_bytes, filename, form_code, form_name, n_errors, n_warnings):
    if history.is_enabled():
        history.record(history.sha256_hex(pdf_bytes), filename,
                       form_code, form_name, n_errors, n_warnings)


@app.route("/review", methods=["POST"])
def review():
    files = request.files.getlist("pdf")
    files = [f for f in files if f and f.filename]
    if not files:
        return render_template("index.html", error="Please choose a PDF file to upload.")
    bad = [f.filename for f in files if not f.filename.lower().endswith(".pdf")]
    if bad:
        return render_template("index.html",
                               error=f"Only PDF files are supported: {', '.join(bad)}")
    # Guard against OOM on the 512MB free tier: cap batch size and file size.
    if len(files) > 10:
        return render_template("index.html",
                               error="Please upload at most 10 files at once.")
    too_big = []
    for f in files:
        f.seek(0, 2)
        if f.tell() > 15 * 1024 * 1024:
            too_big.append(f.filename)
        f.seek(0)
    if too_big:
        return render_template("index.html",
                               error=f"Files over 15 MB are not supported: "
                                     f"{', '.join(too_big)}")
    mode = request.form.get("mode", "auto")
    if mode == "single" or (mode == "auto" and len(files) == 1):
        return _single_report(files[0])
    return _batch_report(files)


def _single_report(f):
    pdf_bytes = f.read()
    already = _dup_check(pdf_bytes, f.filename)
    try:
        result = run(pdf_bytes)
    except ReviewError as exc:
        return render_template("index.html", error=str(exc))
    except Exception:  # noqa: BLE001 - never leak internals to the user
        return render_template("index.html",
                               error="Something went wrong processing that PDF. "
                                     "Please try again or use a different file.")
    if result["form_code"] is None:
        return render_template("index.html", error=result["note"]
                               or "Could not identify the form type.")

    issues = result["issues"]
    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]
    _log_review(pdf_bytes, f.filename, result["form_code"], result["form_name"],
                len(errors), len(warnings))
    pages = []
    for i, ann in enumerate(result["annotated"]):
        # downscale for the browser; keep aspect
        h, w = ann.shape[:2]
        scale = min(1.0, 900 / w)
        if scale < 1.0:
            ann = cv2.resize(ann, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        pages.append({"index": i, "uri": _img_to_data_uri(ann)})
    return render_template(
        "report.html",
        filename=f.filename,
        form_code=result["form_code"],
        form_name=result["form_name"],
        errors=errors,
        warnings=warnings,
        passed=(len(issues) == 0),
        pages=pages,
        already=already,
    )


def _batch_report(files):
    """Run the reviewer on each PDF and render a summary table.

    Files are processed in parallel (2 workers — memory-capped for the
    512MB Render free tier). Large batches are the main OOM risk, so
    file count and size are limited at the route level.
    """
    import gc
    from concurrent.futures import ThreadPoolExecutor

    payloads = [(i, f.filename, f.read()) for i, f in enumerate(files)]

    # All sqlite access stays in the MAIN thread: concurrent writes from
    # worker threads can stall on some filesystems (Render). Workers only
    # run the pipeline; hashing / dup-check / record happen here.
    log_on = history.is_enabled()
    shas, alreadys = [], []
    for _, filename, pdf_bytes in payloads:
        if log_on:
            sha = history.sha256_hex(pdf_bytes)
            shas.append(sha)
            rec = history.find(sha)
            if rec:
                from datetime import datetime
                rec["when"] = datetime.fromtimestamp(
                    rec["reviewed_at"]).strftime("%Y-%m-%d %H:%M")
            alreadys.append(rec)
        else:
            shas.append(None)
            alreadys.append(None)

    def process_one(args):
        idx, filename, pdf_bytes = args
        already = alreadys[idx]
        try:
            r = run(pdf_bytes, annotate_pages=False)
        except ReviewError as exc:
            return idx, {"filename": filename, "ok": False, "error": str(exc),
                         "already": already}
        except Exception:  # noqa: BLE001 - one bad file must not kill the batch
            return idx, {"filename": filename, "ok": False,
                         "error": "Something went wrong processing this file.",
                         "already": already}
        finally:
            del pdf_bytes
            gc.collect()
        if r["form_code"] is None:
            return idx, {"filename": filename, "ok": False,
                         "error": r["note"] or "Could not identify the form type.",
                         "already": already}
        issues = r["issues"]
        errors = [i for i in issues if i["severity"] == "error"]
        warnings = [i for i in issues if i["severity"] == "warning"]
        return idx, {
            "filename": filename,
            "ok": True,
            "form_code": r["form_code"],
            "form_name": r["form_name"],
            "n_errors": len(errors),
            "n_warnings": len(warnings),
            "passed": len(issues) == 0,
            "errors": errors,
            "warnings": warnings,
            "already": already,
            "_sha": shas[idx],
        }

    results = [None] * len(payloads)
    with ThreadPoolExecutor(max_workers=min(2, len(payloads))) as ex:
        for idx, res in ex.map(process_one, payloads):
            results[idx] = res
    gc.collect()
    if log_on:
        for res in results:
            if res.get("ok") and res.get("_sha"):
                history.record(res["_sha"], res["filename"], res["form_code"],
                               res["form_name"], res["n_errors"], res["n_warnings"])
            res.pop("_sha", None)
    n_passed = sum(1 for r in results if r.get("passed"))
    token = uuid.uuid4().hex
    BATCH_CACHE[token] = results
    # keep the cache small
    while len(BATCH_CACHE) > 20:
        BATCH_CACHE.pop(next(iter(BATCH_CACHE)))
    return render_template("batch.html", results=results, n_files=len(results),
                           n_passed=n_passed, csv_token=token)


@app.route("/report/<token>.csv")
def batch_csv(token):
    results = BATCH_CACHE.get(token)
    if not results:
        return render_template("index.html",
                               error="That report has expired — please re-upload."), 410
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["filename", "form", "status", "severity", "field", "message"])
    for r in results:
        if not r.get("ok"):
            w.writerow([r["filename"], "", "error", "", "",
                        r.get("error", "")])
            continue
        status = "PASS" if r["passed"] else "FAIL"
        issues = r["errors"] + r["warnings"]
        if not issues:
            w.writerow([r["filename"], r["form_name"], status, "", "", ""])
        for iss in issues:
            w.writerow([r["filename"], r["form_name"], status,
                        iss["severity"], iss["field"], iss["message"]])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition":
                             "attachment; filename=regenmed-review-report.csv"})


@app.route("/history")
def history_page():
    from datetime import datetime
    entries = history.recent(100)
    for e in entries:
        e["when"] = datetime.fromtimestamp(e["reviewed_at"]).strftime(
            "%Y-%m-%d %H:%M")
    return render_template("history.html", entries=entries,
                           log_enabled=history.is_enabled())


@app.route("/history/toggle", methods=["POST"])
def history_toggle():
    history.set_enabled(not history.is_enabled())
    return redirect(url_for("history_page"))


@app.route("/history/clear", methods=["POST"])
def history_clear():
    history.clear()
    return redirect(url_for("history_page"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
