"""Low-level vision helpers: ink detection, cluster counting, slash/date structure."""
import cv2
import numpy as np

INK_THRESH = 200        # grayscale < this counts as ink
FILL_FRAC = 0.02        # ink fraction above this => cell is filled
MIN_INK_PX = 40         # absolute floor to ignore speckles


def crop_frac(img, box):
    """Crop fractional box (x0,y0,x1,y1) -> image region."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    return img[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]


def ink_mask(img, thresh=INK_THRESH):
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    return b


def ink_fraction(img, box, thresh=INK_THRESH, inset=0.12):
    """Fraction of dark pixels inside box after shrinking by inset (avoid borders)."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    ix0 = int((x0 + (x1 - x0) * inset) * W)
    ix1 = int((x1 - (x1 - x0) * inset) * W)
    iy0 = int((y0 + (y1 - y0) * inset) * H)
    iy1 = int((y1 - (y1 - y0) * inset) * H)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    region = img[iy0:iy1, ix0:ix1]
    gray = region if len(region.shape) == 2 else cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return float((gray < thresh).mean())


def is_filled(img, box, thresh=INK_THRESH, inset=0.12, frac=FILL_FRAC):
    if ink_fraction(img, box, thresh, inset) < frac:
        return False
    # absolute pixel-count floor against speckles
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    area = (x1 - x0) * W * (y1 - y0) * H * (1 - 2 * inset) ** 2
    return ink_fraction(img, box, thresh, inset) * area > MIN_INK_PX


def count_x_clusters(img, box, thresh=170, min_gap_px=8):
    """Count ink clusters separated along x (e.g. initials + date written side by side)."""
    region = crop_frac(img, box)
    gray = region if len(region.shape) == 2 else cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    col_ink = (gray < thresh).mean(axis=0)
    ink_cols = np.where(col_ink > 0.03)[0]
    if len(ink_cols) == 0:
        return 0
    clusters = 1
    for a, b in zip(ink_cols[:-1], ink_cols[1:]):
        if b - a > min_gap_px:
            clusters += 1
    return clusters


def count_y_clusters(img, box, thresh=170, min_gap_frac=0.10):
    """Count ink clusters separated along y (e.g. initials stacked over date)."""
    region = crop_frac(img, box)
    H = region.shape[0]
    gray = region if len(region.shape) == 2 else cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    row_ink = (gray < thresh).mean(axis=1)
    ink_rows = np.where(row_ink > 0.03)[0]
    if len(ink_rows) == 0:
        return 0
    min_gap = max(3, int(H * min_gap_frac))
    clusters = 1
    for a, b in zip(ink_rows[:-1], ink_rows[1:]):
        if b - a > min_gap:
            clusters += 1
    return clusters


def has_diagonal_slash(img, box, thresh=175):
    """Detect a long diagonal stroke (the '/' in handwritten 'N/A').

    The slash must be long AND vertically centered in the box — this
    distinguishes the centered 'N/A' slash from '/' separators inside a
    date (which sit in the lower part of the cell) and from short
    diagonal digit strokes.
    """
    region = crop_frac(img, box)
    H, W = region.shape[:2]
    gray = region if len(region.shape) == 2 else cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(b, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=12,
                            minLineLength=int(0.30 * H), maxLineGap=6)
    if lines is None:
        return False
    for l in lines.reshape(-1, 4):
        x1, y1, x2, y2 = (int(v) for v in l)
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx == 0:
            continue
        ang = float(np.degrees(np.arctan2(dy, dx)))
        ymid = (y1 + y2) / 2 / H
        if (25 <= ang <= 70 and np.hypot(dx, dy) > 0.30 * H
                and 0.30 < ymid < 0.70):
            return True
    return False


def ocr_text(img, box, config="--psm 7"):
    """Best-effort OCR of a small region. Returns stripped text (may be garbage)."""
    try:
        from pytesseract import image_to_string
    except ImportError:
        return ""
    region = crop_frac(img, box)
    if region.size == 0:
        return ""
    big = cv2.resize(region, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return image_to_string(big, config=config).strip()
