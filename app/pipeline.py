"""Pipeline: PDF -> page images -> deskew -> classify -> check -> annotate."""
import io
import cv2
import numpy as np
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError
from pytesseract import image_to_string

from .forms import FORMS
from .checks import CHECKERS
from .vision import crop_frac

RENDER_DPI = 150
MAX_PAGES = 10  # forms are 1-2 pages; anything more is user error / DoS


class ReviewError(Exception):
    """A user-friendly processing failure (shown as-is, no traceback)."""


def render_pdf(pdf_bytes):
    """Render PDF pages to grayscale numpy arrays.

    Raises ReviewError with a user-friendly message for corrupt,
    encrypted, empty, or oversized PDFs.
    """
    if not pdf_bytes or len(pdf_bytes) < 200:
        raise ReviewError("The file is empty or too small to be a valid PDF.")
    if not pdf_bytes[:5].startswith(b"%PDF"):
        raise ReviewError("This doesn't look like a PDF file — please upload a valid PDF.")
    try:
        pil_pages = convert_from_bytes(pdf_bytes, dpi=RENDER_DPI, grayscale=True)
    except (PDFPageCountError, PDFSyntaxError):
        raise ReviewError("Could not read this PDF — the file appears to be corrupted.")
    except Exception as exc:
        msg = str(exc).lower()
        if "password" in msg or "encrypt" in msg:
            raise ReviewError("This PDF is password-protected. Please upload an unlocked copy.")
        raise ReviewError(f"Could not read this PDF ({exc}).")
    if not pil_pages:
        raise ReviewError("This PDF has no readable pages.")
    truncated = False
    if len(pil_pages) > MAX_PAGES:
        pil_pages = pil_pages[:MAX_PAGES]
        truncated = True
    pages = [np.array(p) for p in pil_pages]
    return pages, truncated


def deskew(img):
    """Correct small scan rotation from the form's own printed rules.

    The previous min-area-rect-of-all-ink approach was fragile: a blob of
    handwriting could swing the angle several degrees and actively tilt an
    otherwise straight scan, misaligning every calibrated box downstream.
    Long printed table rules are the ground truth for the form's orientation;
    the median Hough-line angle is robust to handwriting and short strokes.
    Falls back to no rotation when too few long lines are found.
    """
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=300,
                            minLineLength=int(W * 0.4), maxLineGap=10)
    if lines is None:
        return img
    angs = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        a = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if a > 90:
            a -= 180
        elif a < -90:
            a += 180
        if abs(a) < 8:  # near-horizontal rule
            angs.append(a)
    if len(angs) < 3:
        return img
    angle = float(np.median(angs))
    if abs(angle) > 5 or abs(angle) < 0.15:  # only fix small, real skew
        return img
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _ocr_region(img, box):
    region = crop_frac(img, box)
    big = cv2.resize(region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return big


def classify(pages):
    """Identify the form from footer codes + title text. Returns (code, confidence).

    Footer and title regions are stacked into a single image for one
    Tesseract call instead of two (~174ms saved per classify).
    """
    img = pages[0]
    footer = _ocr_region(img, (0.55, 0.94, 1.0, 1.0))
    title = _ocr_region(img, (0.10, 0.02, 0.90, 0.09))
    # stack vertically (pad widths to match)
    w = max(footer.shape[1], title.shape[1])
    def pad(a):
        if a.shape[1] < w:
            padw = w - a.shape[1]
            a = cv2.copyMakeBorder(a, 0, 0, 0, padw, cv2.BORDER_CONSTANT,
                                   value=255)
        return a
    combined = np.vstack([pad(title), pad(footer)])
    hay = image_to_string(combined).upper()
    for code, t in FORMS.items():
        for marker in t["footer_markers"]:
            if marker.upper() in hay:
                return code, "footer"
    for code, t in FORMS.items():
        for marker in t["title_markers"]:
            if marker.upper() in hay:
                return code, "title"
    # lot log page 2 has no footer on p1? try page 2 footer too
    if len(pages) > 1:
        footer2 = image_to_string(
            _ocr_region(pages[1], (0.55, 0.94, 1.0, 1.0))).upper()
        for code, t in FORMS.items():
            for marker in t["footer_markers"]:
                if marker.upper() in footer2:
                    return code, "footer"
    return None, None


def annotate(img, issues, page_idx):
    """Draw issue boxes on a copy of the page; returns BGR image."""
    out = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out = out.copy()
    H, W = out.shape[:2]
    for iss in issues:
        if iss["page"] != page_idx or not iss["box"]:
            continue
        x0, y0, x1, y1 = iss["box"]
        p0 = (int(x0 * W), int(y0 * H))
        p1 = (int(x1 * W), int(y1 * H))
        color = (0, 0, 255) if iss["severity"] == "error" else (0, 165, 255)
        cv2.rectangle(out, p0, p1, color, 3)
    return out


def run(pdf_bytes, annotate_pages=True, filename=""):
    """Full review pipeline.

    annotate_pages=False skips drawing issue boxes (batch mode never
    displays them — saves a full-page copy + drawing per page).
    """
    pages, truncated = render_pdf(pdf_bytes)
    try:
        pages = [deskew(p) for p in pages]
        code, how = classify(pages)
    except Exception as exc:  # OCR / image failures -> friendly, not a 500
        raise ReviewError(f"Could not analyze this PDF ({exc}).")
    if code is None:
        return {"form_code": None, "form_name": None, "issues": [],
                "pages": len(pages), "annotated": [pages] if annotate_pages else [],
                "note": "Could not identify the form type — is this one of the "
                        "four supported RegenMed forms?"}
    issues = CHECKERS[code](pages)
    if truncated:
        issues.append({"severity": "warning", "field": "Document",
                       "message": f"Only the first {MAX_PAGES} pages were reviewed.",
                       "box": None, "page": 0})
    annotated = [annotate(p, issues, i) for i, p in enumerate(pages)] \
        if annotate_pages else []
    result = {"form_code": code, "form_name": FORMS[code]["name"],
              "issues": issues, "pages": len(pages), "annotated": annotated,
              "note": None, "ai_summary": None}
    # Optional Gemini layer (text-only): consistency findings + plain-language
    # summary. Silent no-op unless HACKATHON_API_KEY is set; never breaks
    # the geometric review.
    try:
        from app.ai_review import enhance
        enhance(result, pages, filename)
    except Exception:  # noqa: BLE001
        pass
    return result
