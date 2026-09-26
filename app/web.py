"""Flask web app: upload a processing-form PDF, get a validation report."""
import base64
import io
import os

import cv2
from flask import Flask, render_template, request

from .pipeline import run

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


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
    if len(files) == 1:
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
    """Run the reviewer on each PDF and render a summary table."""
    results = []
    for f in files:
        pdf_bytes = f.read()
        try:
            r = run(pdf_bytes)
        except Exception as exc:  # noqa: BLE001
            results.append({"filename": f.filename, "ok": False,
                            "error": f"Could not process ({exc})."})
            continue
        if r["form_code"] is None:
            results.append({"filename": f.filename, "ok": False,
                            "error": r["note"] or "Could not identify the form type."})
            continue
        issues = r["issues"]
        errors = [i for i in issues if i["severity"] == "error"]
        warnings = [i for i in issues if i["severity"] == "warning"]
        results.append({
            "filename": f.filename,
            "ok": True,
            "form_code": r["form_code"],
            "form_name": r["form_name"],
            "n_errors": len(errors),
            "n_warnings": len(warnings),
            "passed": len(issues) == 0,
            "errors": errors,
            "warnings": warnings,
        })
    n_passed = sum(1 for r in results if r.get("passed"))
    return render_template("batch.html", results=results, n_files=len(results),
                           n_passed=n_passed)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
