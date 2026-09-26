"""Pipeline: PDF -> page images -> deskew -> classify -> check -> annotate."""
import io
import cv2
import numpy as np
from pdf2image import convert_from_bytes
from pytesseract import image_to_string

from .forms import FORMS
from .checks import CHECKERS
from .vision import crop_frac

RENDER_DPI = 150


def render_pdf(pdf_bytes):
    """Render PDF pages to grayscale numpy arrays."""
    pil_pages = convert_from_bytes(pdf_bytes, dpi=RENDER_DPI, grayscale=True)
    return [np.array(p) for p in pil_pages]


def deskew(img):
    """Correct small scan rotation via min-area-rect of ink."""
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    coords = np.column_stack(np.where(b > 0))
    if len(coords) < 100:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) > 5:  # only fix small skew
        return img
    H, W = gray.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _ocr_region(img, box):
    region = crop_frac(img, box)
    big = cv2.resize(region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return image_to_string(big).upper()


def classify(pages):
    """Identify the form from footer codes + title text. Returns (code, confidence)."""
    img = pages[0]
    H, W = img.shape[:2]
    footer = _ocr_region(img, (0.55, 0.94, 1.0, 1.0))
    title = _ocr_region(img, (0.10, 0.02, 0.90, 0.09))
    hay = footer + " " + title
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
        footer2 = _ocr_region(pages[1], (0.55, 0.94, 1.0, 1.0)).upper()
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


def run(pdf_bytes):
    pages = [deskew(p) for p in render_pdf(pdf_bytes)]
    code, how = classify(pages)
    if code is None:
        return {"form_code": None, "form_name": None, "issues": [],
                "pages": len(pages), "annotated": [pages],
                "note": "Could not identify the form type."}
    issues = CHECKERS[code](pages)
    annotated = [annotate(p, issues, i) for i, p in enumerate(pages)]
    return {"form_code": code, "form_name": FORMS[code]["name"],
            "issues": issues, "pages": len(pages), "annotated": annotated,
            "note": None}
