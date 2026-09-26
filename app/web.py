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
    f = request.files.get("pdf")
    if f is None or not f.filename:
        return render_template("index.html", error="Please choose a PDF file to upload.")
    if not f.filename.lower().endswith(".pdf"):
        return render_template("index.html", error="Only PDF files are supported.")
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
