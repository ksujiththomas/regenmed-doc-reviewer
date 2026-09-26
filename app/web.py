"""Flask web app: upload a processing-form PDF, get a validation report."""
import base64
import io
import os

import cv2
from flask import Flask, render_template, request

from .pipeline import run

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 160 * 1024 * 1024  # 10 files x 15 MB + overhead


@app.errorhandler(413)
def too_large(_):
    return render_template("index.html",
                           error="Total upload is too large — please keep it under "
                                 "150 MB (max 10 files, 15 MB each)."), 413


def _img_to_data_uri(bgr_img):
    ok, buf = cv2.imencode(".png", bgr_img)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


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
    try:
        result = run(pdf_bytes)
    except Exception as exc:  # noqa: BLE001 - show friendly error
        return render_template("index.html",
                               error=f"Could not process that PDF ({exc}).")
    if result["form_code"] is None:
        return render_template("index.html", error=result["note"]
                               or "Could not identify the form type.")

    issues = result["issues"]
    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]
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
    )


def _batch_report(files):
    """Run the reviewer on each PDF and render a summary table.

    Files are processed in parallel (2 workers — memory-capped for the
    512MB Render free tier). Large batches are the main OOM risk, so
    file count and size are limited at the route level.
    """
    import gc
    from concurrent.futures import ThreadPoolExecutor

    def process_one(args):
        idx, filename, pdf_bytes = args
        try:
            r = run(pdf_bytes)
        except Exception as exc:  # noqa: BLE001
            return idx, {"filename": filename, "ok": False,
                         "error": f"Could not process ({exc})."}
        finally:
            del pdf_bytes
            gc.collect()
        if r["form_code"] is None:
            return idx, {"filename": filename, "ok": False,
                         "error": r["note"] or "Could not identify the form type."}
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
        }

    payloads = [(i, f.filename, f.read()) for i, f in enumerate(files)]
    results = [None] * len(payloads)
    with ThreadPoolExecutor(max_workers=min(2, len(payloads))) as ex:
        for idx, res in ex.map(process_one, payloads):
            results[idx] = res
    gc.collect()
    n_passed = sum(1 for r in results if r.get("passed"))
    return render_template("batch.html", results=results, n_files=len(results),
                           n_passed=n_passed)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
