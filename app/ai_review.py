"""Gemini-powered review layer for the document pipeline.

Two jobs, one API request per form:
  1. Consistency checker — cross-field logic on best-effort OCR reads
     (date chronology, donor-number sanity, patterns across issues).
  2. Report writer — a plain-language summary of the review.

Text-only: the hackathon Gemini proxy takes no images, so this layer works
from the geometric pipeline's issue list plus noisy OCR reads of a few key
fields. Findings are always warnings (never errors) — the AI advises, the
geometric checks decide pass/fail.

Inactive unless the HACKATHON_API_KEY environment variable is set; any
failure (timeout, quota, bad response) degrades silently to no AI review.
"""

import json
import logging
import os

import requests

from app.forms import MP_F_018, MP_F_023
from app.vision import ocr_text

log = logging.getLogger(__name__)

API_URL = ("https://hackathon-api-new-152590733511.northamerica-northeast2.run.app"
           "/api/generate")
TIMEOUT = 60  # seconds; README example uses 90

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["field", "message"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["findings", "summary"],
}

DATE_OCR = "--psm 7 -c tessedit_char_whitelist=0123456789/-"
DONOR_OCR = "--psm 7 -c tessedit_char_whitelist=0123456789"


def _clean(s, limit=40):
    s = " ".join(str(s).split())
    return s[:limit]


def _val(img, box, config):
    """One extracted value: best-effort OCR read plus the geometric
    handwriting verdict. The read may be '' or garbled even when handwriting
    is present — callers must never treat an empty read as a blank field
    when detected is True."""
    from app.vision import is_filled
    return {"read": _clean(ocr_text(img, box, config)),
            "handwriting_detected": bool(is_filled(img, box))}


def extract_values(pages, form_code):
    """Best-effort OCR reads of key fields, each paired with the geometric
    handwriting verdict so the AI cannot mistake a bad OCR read for a
    blank field."""
    values = {}
    img = pages[0]
    try:
        if form_code == "MP-F-023":
            t = MP_F_023
            values["donor_number"] = _val(img, t["donor_value_box"], DONOR_OCR)
            y0, y1 = t["header_value_y"]
            for label in t["date_fields"]:
                cx0, cx1 = t["header_cols"][label]
                values[label] = _val(img, (cx0, y0, cx1, y1), DATE_OCR)
        elif form_code == "MP-F-018":
            t = MP_F_018
            values["donor_number"] = _val(img, t["donor_value_box"], DONOR_OCR)
            for key, label in (("discard_date_box", "Discard date"),
                               ("released_date_box", "Released date"),
                               ("donorchart_date_box", "Donor chart date")):
                values[label] = _val(img, t[key], DATE_OCR)
    except Exception:  # noqa: BLE001 - extraction is best-effort
        log.warning("AI value extraction failed", exc_info=True)
    return values


def _prompt(form_code, form_name, filename, issues, values):
    slim_issues = [{"field": i.get("field"), "severity": i.get("severity"),
                    "message": i.get("message")} for i in issues]
    return f"""You are a QA reviewer for RegenMed tissue-processing forms. A geometric checker has already verified that required fields are present; your job is (a) cross-field consistency and (b) a plain-language summary.

Form: {form_name} ({form_code}), file: {filename}

Extracted field values (each has a best-effort OCR "read" of the handwriting
plus "handwriting_detected" from a geometric ink check — the read is noisy and
may be empty or garbled even when handwriting IS present):
{json.dumps(values, indent=1)}

HARD RULES:
- NEVER claim a field is blank, missing, or "not recorded" when its
  handwriting_detected is true. An empty read with handwriting_detected=true
  means the OCR failed, not that the field is empty.
- NEVER restate an issue already listed in the geometric check results below —
  only NEW consistency findings.
- NEVER guess or invent a value. Ignore any reading that is not clearly
  interpretable.

Geometric check results:
{json.dumps(slim_issues, indent=1)}

Tasks:
1. Consistency findings: check date chronology (e.g. Date of Recovery should not be after Date of Processing; discard/release dates should be sensible relative to each other), donor-number format sanity, and patterns across the issues above (e.g. several missing sign-offs on one form suggesting it was never completed). Report ONLY findings you are confident about. When the OCR reads are too noisy to judge, report nothing.
2. Summary: 2-4 sentences in plain language stating what was checked, the overall state of the form, and the single most important action if anything needs attention. If there are no issues, say the form is complete.

Return JSON with keys "findings" (array of {{"field", "message"}}) and "summary" (string)."""


def ai_review(form_code, form_name, filename, issues, values):
    """One Gemini request. Returns {"findings": [...], "summary": str}
    or None on any failure."""
    api_key = os.environ.get("HACKATHON_API_KEY")
    if not api_key:
        return None
    try:
        resp = requests.post(
            API_URL,
            headers={"X-API-Key": api_key, "Content-Type": "application/json"},
            json={"contents": _prompt(form_code, form_name, filename,
                                      issues, values),
                  "response_schema": RESPONSE_SCHEMA},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        log.info("Gemini requests_remaining=%s", data.get("requests_remaining"))
        parsed = json.loads(data["text"])
        findings = parsed.get("findings") or []
        summary = parsed.get("summary") or ""
        # sanitize: keep it small and well-formed
        clean = []
        for f in findings[:10]:
            if isinstance(f, dict) and f.get("message"):
                clean.append({"field": _clean(f.get("field") or "General", 60),
                              "message": _clean(f["message"], 300)})
        return {"findings": clean, "summary": _clean(summary, 1200)}
    except Exception:  # noqa: BLE001 - AI is advisory; never break the pipeline
        log.warning("Gemini review failed", exc_info=True)
        return None


def enhance(result, pages, filename):
    """Run the AI layer over a pipeline result dict in place. Silent no-op
    unless HACKATHON_API_KEY is set and the call succeeds."""
    if result.get("form_code") is None:
        return result
    if not os.environ.get("HACKATHON_API_KEY"):
        return result
    values = extract_values(pages, result["form_code"])
    review = ai_review(result["form_code"], result.get("form_name"),
                       filename, result.get("issues", []), values)
    if not review:
        return result
    for f in review["findings"]:
        result["issues"].append({
            "severity": "warning",
            "field": "AI review — " + f["field"],
            "message": f["message"],
            "box": None,
            "page": 0,
            "ai": True,
        })
    result["ai_summary"] = review["summary"]
    return result
