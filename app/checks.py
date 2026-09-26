"""Per-form validation rule engines. Each returns a list of issue dicts:

    {"severity": "error"|"warning", "field": str, "message": str,
     "box": (x0,y0,x1,y1) | None, "page": int}
"""
import re
import cv2
import numpy as np

from .vision import (is_filled, ink_fraction, count_x_clusters, count_y_clusters,
                     has_diagonal_slash, ocr_text)
from .forms import MP_F_023, QS_F_049, LOT_LOG, MP_F_018

DATE_RE = re.compile(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2}(\d{2})?$")
# Confident wrong-format read: YYYY-MM-DD / YYYY/MM/DD. Only a full,
# separator-delimited 4-digit-year read counts — digit fragments from
# noisy OCR ("2/29") must NOT be flagged, they fall through to the
# structural check instead.
WRONG_FMT_RE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$")
NA_RE = re.compile(r"^\W*n\W*a\W*$", re.IGNORECASE)


def _issue(severity, field, message, box=None, page=0):
    return {"severity": severity, "field": field, "message": message,
            "box": box, "page": page}


def _ink_x_span(img, box, thresh=170):
    """Fraction of page width spanned by ink columns inside box."""
    from .vision import crop_frac
    region = crop_frac(img, box)
    H, Wfull = img.shape[:2]
    gray = region if len(region.shape) == 2 else cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    col_ink = (gray < thresh).mean(axis=0)
    ink_cols = np.where(col_ink > 0.03)[0]
    if len(ink_cols) == 0:
        return 0.0
    return float(ink_cols[-1] - ink_cols[0]) / Wfull  # fraction of page width


def check_date_zone(img, box, field, page=0, strict_format=False):
    """Date must be present; QS-F-049 additionally flags a clear MM/DD/YY violation.

    strict_format=False (MP-F-023, MP-F-018 — the brief only requires a date
    be present): fast structural pass first (~50ms vs ~174ms for a Tesseract
    subprocess); OCR only runs when the structure is ambiguous.
    strict_format=True (QS-F-049 — the brief mandates MM/DD/YY): a confident
    OCR read of a wrong format (e.g. YYYY-MM-DD) is an error. Messy
    handwriting that OCR cannot parse is accepted silently — flagging it
    would be a false alarm, and no structural separator check proved
    reliable on real handwriting.
    """
    if not is_filled(img, box):
        return [_issue("error", field, "Date is blank.", box, page)]

    def ocr_digits():
        t = ocr_text(img, box, "--psm 7 -c tessedit_char_whitelist=0123456789/-")
        return t.replace(" ", "").strip("/-")

    if strict_format:
        if WRONG_FMT_RE.match(ocr_digits()):
            return [_issue("error", field,
                           "Date format looks invalid (expected MM/DD/YY).", box, page)]
        return []
    if _looks_like_date(img, box):
        return []  # fast path: clearly a date
    if DATE_RE.match(ocr_digits()):
        return []
    # structural fallback: too many strokes => probably words, not a date
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in (box[0] * W, box[1] * H, box[2] * W, box[3] * H)]
    c = img[y0:y1, x0:x1]
    gray = c if len(c.shape) == 2 else cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    comps = sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 25)
    if comps > 12:
        return [_issue("error", field,
                       "Date format looks invalid (expected MM/DD/YY).", box, page)]
    return [_issue("warning", field,
                   "Date is present but the MM/DD/YY format could not be "
                   "machine-verified — please confirm visually.", box, page)]


def _looks_like_date(img, box):
    """Structural check: does the ink in box look like a handwritten date?
    A date (MM-DD-YY) is a wide region with several digit-sized blobs.
    Grid lines are removed morphologically first.
    """
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in (box[0] * W, box[1] * H, box[2] * W, box[3] * H)]
    c = img[y0:y1, x0:x1]
    if c.size == 0:
        return False
    gray = c if len(c.shape) == 2 else cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    # remove grid lines
    hkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    lines = cv2.morphologyEx(b, cv2.MORPH_OPEN, hkernel)
    lines = cv2.bitwise_or(lines, cv2.morphologyEx(b, cv2.MORPH_OPEN, vkernel))
    ink = cv2.bitwise_and(b, cv2.bitwise_not(lines))
    # must be wide (dates are wider than tall)
    ys, xs = np.where(ink > 0)
    if len(xs) < 50:
        return False
    wpx, hpx = xs.max() - xs.min(), ys.max() - ys.min()
    if wpx < 1.2 * hpx:
        return False  # not wide enough for a date
    # must have several digit-sized components (not a single scribble)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    comps = sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 15)
    return 4 <= comps <= 20


def _is_shaded(img, box):
    """Gray-shaded cell (N/A by design) vs white writable cell.

    Shaded fill reads ~180, white paper ~254; ink (<150) is excluded.
    """
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in (box[0] * W, box[1] * H, box[2] * W, box[3] * H)]
    c = img[y0:y1, x0:x1]
    if c.size == 0:
        return False
    gray = c if len(c.shape) == 2 else cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    bright = gray[gray > 150]
    if bright.size == 0:
        return False
    return float(np.median(bright)) < 220


def _find_frfd_centers(img):
    """Y-centers of the 17 MP-F-023 data rows from the printed FRZ/FD text.

    FRZ/FD is pre-printed in every data row; blank spacer rows have none.
    Returns [] if detection fails (caller falls back to hardcoded centers).
    """
    H, W = img.shape[:2]
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x0, x1 = int(0.335 * W), int(0.415 * W)
    region = gray[int(0.48 * H):int(0.84 * H), x0:x1]
    _, b = cv2.threshold(region, 170, 255, cv2.THRESH_BINARY_INV)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    centers = []
    for i in range(1, n):
        area, wpx, hpx = (stats[i, cv2.CC_STAT_AREA], stats[i, cv2.CC_STAT_WIDTH],
                          stats[i, cv2.CC_STAT_HEIGHT])
        if area > 200 and 15 < wpx < 120 and 8 < hpx < 40:
            yc = (stats[i, cv2.CC_STAT_TOP] + hpx / 2 + int(0.48 * H)) / H
            if yc > 0.49:  # skip the "FRZ / FD" column-header text
                centers.append(yc)
    centers.sort()
    return centers if len(centers) == 17 else []


def _qty_cell_filled(img, yc, x0f, x1f):
    """Is a # Produced / # Packaged cell filled?

    The FRZ/FD print center (yc) anchors a fixed window covering the cell
    interior; grid-line halo rows are excluded. A cell counts as filled if
    it contains a coherent ink stroke (a handwritten digit is >=6px tall at
    200dpi; grid-line residue forms flat fragments <=2px tall).
    """
    H, W = img.shape[:2]
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x0, x1 = int((x0f + 0.010) * W), int((x1f - 0.010) * W)
    y0, y1 = int((yc - 0.007) * H), int((yc + 0.007) * H)
    c = gray[y0:y1, x0:x1]
    if c.size == 0:
        return False
    dark = (c < 200)
    rowfrac = dark.mean(axis=1)
    keep = rowfrac < 0.35
    if keep.sum() < 3:
        return False
    m = np.where(dark[keep], 255, 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_HEIGHT] >= 6:
            return True
    return False


def _has_initials(img, box):
    """Do initials (a handwritten stroke) appear in the box?

    Line-aware: vertical/horizontal grid-line fragments are ignored, so
    neighboring cells' dividers or spillover don't count.
    """
    H, W = img.shape[:2]
    x0, y0, x1, y1 = (int(box[0] * W), int(box[1] * H),
                      int(box[2] * W), int(box[3] * H))
    c = img[y0:y1, x0:x1]
    if c.size == 0:
        return False
    gray = c if len(c.shape) == 2 else cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    dark = np.where(gray < 200, 255, 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    for i in range(1, n):
        a, hh, ww = (stats[i, cv2.CC_STAT_AREA], stats[i, cv2.CC_STAT_HEIGHT],
                     stats[i, cv2.CC_STAT_WIDTH])
        if ww <= 4 and hh >= 12:
            continue  # vertical divider/fragment
        if hh <= 4 and ww >= 12:
            continue  # horizontal line fragment
        if hh >= c.shape[0] * 0.85 or ww >= c.shape[1] * 0.85:
            continue  # spans the box: a rule line, not handwriting
        if hh >= 10:
            return True
    return False


def _date_dashes(img, box):
    """X-positions of dash-like components (the '-' in MM-DD-YY dates)."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = (int(box[0] * W), int(box[1] * H),
                      int(box[2] * W), int(box[3] * H))
    c = img[y0:y1, x0:x1]
    if c.size == 0:
        return []
    gray = c if len(c.shape) == 2 else cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    dark = np.where(gray < 200, 255, 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    out = []
    for i in range(1, n):
        wpx, hpx = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if 8 <= wpx <= 30 and hpx <= 7 and wpx >= 2 * hpx:
            out.append(stats[i, cv2.CC_STAT_LEFT] / W + box[0])
    return sorted(out)


# ------------------------------------------------------------- MP-F-023
def check_mp_f023(pages):
    t = MP_F_023
    img = pages[0]
    issues = []
    y0, y1 = t["header_value_y"]

    for label, (cx0, cx1) in t["header_cols"].items():
        box = (cx0, y0, cx1, y1)
        if label in t["stacked_cells"]:
            cx0, cx1, ix0, ix1 = t["stacked_cells"][label]
            top, mid, bot = t["by_date_split"]
            ibox, dbox = (ix0, top, ix1, mid), (cx0, mid, cx1, bot)
            if not _has_initials(img, ibox):
                issues.append(_issue("error", f"{label} — By (initials)",
                                     "Initials are missing.", ibox))
            issues.extend(check_date_zone(img, dbox, f"{label} — Date"))
        elif label == "Donor #":
            vbox = t["donor_value_box"]
            if not is_filled(img, vbox):
                issues.append(_issue("error", "Donor #", "Donor # is blank.", vbox))
        else:
            if not is_filled(img, box):
                issues.append(_issue("error", label, f"{label} is blank.", box))
            elif label in t["date_fields"]:
                issues.extend(check_date_zone(img, box, label))

    # "Verified By" cell under Donor #: part of the required header block.
    vbox = t["verified_by_box"]
    if not is_filled(img, vbox):
        issues.append(_issue("error", "Verified By",
                             "Verified By initials are blank.", vbox))

    # Operations Manager Review: needs initials AND date ("Initials / Date").
    # The date's dashes anchor the search: initials must appear as a
    # handwritten stroke in the zone left of the first dash. A date alone
    # (with its separated digits) must not pass as "initials + date".
    obox = t["ops_review_box"]
    dashes = _date_dashes(img, obox)
    if not dashes:
        n = count_x_clusters(img, obox)
        if n == 0:
            issues.append(_issue("error", "Operations Manager Review",
                                 "Review initials/date are missing.", obox))
        else:
            issues.append(_issue("warning", "Operations Manager Review",
                                 "Review entry is unclear — verify initials and date are both present.",
                                 obox))
    else:
        ibox = (0.535, obox[1], min(dashes) - 0.020, obox[3])
        if not _has_initials(img, ibox):
            issues.append(_issue("error", "Operations Manager Review",
                                 "Review initials are missing (date present, initials blank).",
                                 ibox))

    # Tissue rows: for white (non-shaded) cells, a listed tissue needs
    # BOTH # Produced and # Packaged filled. Shaded cells are N/A by design.
    # Tissue names are pre-printed, so the 17 data rows are located via the
    # printed FRZ/FD text (blank spacer rows have none). Column edges measured
    # from the printed grid: produced 0.475-0.578, packaged 0.578-0.681.
    row_centers = _find_frfd_centers(img) or t["frfd_fallback_centers"]
    for i, yc in enumerate(row_centers):
        label = t["tissue_names"][i] if i < len(t["tissue_names"]) else f"Tissue row {i + 1}"
        pbox = (0.475 + 0.010, yc - 0.007, 0.578 - 0.010, yc + 0.007)
        kbox = (0.578 + 0.010, yc - 0.007, 0.681 - 0.010, yc + 0.007)
        if not _qty_cell_filled(img, yc, 0.475, 0.578) and not _is_shaded(img, pbox):
            issues.append(_issue("error", label, "# Produced is blank.", pbox))
        if not _qty_cell_filled(img, yc, 0.578, 0.681) and not _is_shaded(img, kbox):
            issues.append(_issue("error", label, "# Packaged is blank.", kbox))
    return issues


# ------------------------------------------------------------- QS-F-049
def _bottom_ink_width_frac(img, box):
    """Total width of ink components as a fraction of box width."""
    from .vision import crop_frac
    region = crop_frac(img, box)
    if region.size == 0:
        return 0.0
    gray = region if len(region.shape) == 2 else cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    wsum = sum(stats[i, cv2.CC_STAT_WIDTH] for i in range(1, n)
               if stats[i, cv2.CC_STAT_AREA] >= 25)
    return wsum / region.shape[1]


def _is_na(img, cell, dbox):
    """Explicit 'N/A': centered long slash + narrow (non-date-like) bottom."""
    return (has_diagonal_slash(img, cell)
            and _bottom_ink_width_frac(img, dbox) < 0.65)


def _cell_zones(t, band, col):
    y0, y1 = band
    x0, x1 = col
    # writers often put initials right on/above the top border: extend up a touch
    top = max(0.0, y0 - 0.010)
    split = y0 + (y1 - y0) * t["initials_frac"]
    return (x0, top, x1, split), (x0, split, x1, y1), (x0, y0, x1, y1)


def check_qs_f049(pages):
    t = QS_F_049
    img = pages[0]
    issues = []
    for i, band in enumerate(t["row_bands"]):
        for colname, col in (("Technical", t["tech_x"]), ("Quality", t["qual_x"])):
            ibox, dbox, cell = _cell_zones(t, band, col)
            field = f"Row {i + 1} — {colname}"
            # explicit N/A (slash mark) satisfies the whole cell
            if _is_na(img, cell, dbox) or NA_RE.match(ocr_text(img, cell)):
                continue
            # Tolerant detection: search a margin-expanded zone for handwriting.
            # The zones are vertically separate (initials top, date bottom),
            # so position distinguishes them; the expanded search tolerates
            # writing slightly outside the printed cell.
            sbox = _expand(cell, 0.012)
            ini = _has_handwriting(img, (sbox[0], sbox[1], sbox[2], ibox[3]))
            dat = _has_handwriting(img, (sbox[0], dbox[1], sbox[2], sbox[3]))
            if ini and dat:
                # Use the expanded date zone for format verification:
                # handwriting often spills outside the printed cell, and a
                # narrow crop cuts off digits so OCR fails.
                wide_dbox = (sbox[0], dbox[1], sbox[2], sbox[3])
                issues.extend(check_date_zone(img, wide_dbox, f"{field} date",
                                              strict_format=True))
            elif not ini and not dat:
                issues.append(_issue("error", field,
                                     "Both initials and date are blank.", cell))
            elif ini and not dat:
                issues.append(_issue("error", f"{field} date",
                                     "Initials are present but the date is missing "
                                     "(or the row is not marked N/A).", dbox))
            else:
                issues.append(_issue("error", f"{field} initials",
                                     "Date is present but initials are missing "
                                     "(or the row is not marked N/A).", ibox))
    # Row 10 INC / Status linkage
    if is_filled(img, t["inc_box"]) and not is_filled(img, t["status_box"]):
        issues.append(_issue("error", "Disposition — Status",
                             "An INC # is recorded but the Status is blank.",
                             t["status_box"]))
    return issues


# ------------------------------------------------------------- Lot Log
def _row_boxes(cols, y0, y1):
    return {k: (v[0], y0, v[1], y1) for k, v in cols.items()}


def _struck_through(img, box):
    """A long hand-drawn stroke through the row's item cell => voided row.

    Finds the longest continuous horizontal dark run (grid lines masked).
    A strike-through is an unbroken run spanning much of the cell width;
    printed/handwritten item names have gaps between characters.
    Call with the row's ITEM cell box.
    """
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in (box[0] * W, box[1] * H, box[2] * W, box[3] * H)]
    band = img[y0:y1, x0:x1].copy()
    if band.size == 0:
        return False
    band[:4, :] = 255
    band[-4:, :] = 255  # mask grid lines
    _, b = cv2.threshold(band, 190, 255, cv2.THRESH_BINARY_INV)
    best = 0
    for r in b:
        d = np.diff(np.concatenate([[0], (r > 0).astype(np.int8), [0]]))
        s = np.where(d == 1)[0]
        e = np.where(d == -1)[0]
        if len(s):
            best = max(best, int((e - s).max()))
    return best > 0.15 * (x1 - x0)


def check_lot_log(pages):
    t = LOT_LOG
    issues = []
    img1 = pages[0]

    # page 1: item table (skip header band). Tolerant handwriting detection
    # absorbs slight misalignment and ignores grid-line residue. Search
    # boxes are margin-expanded: writers often spill into neighboring
    # columns (e.g. a large "N/A").
    cols = t["p1_cols"]
    for j in range(1, len(t["p1_hlines"]) - 1):
        y0, y1 = t["p1_hlines"][j] + 0.001, t["p1_hlines"][j + 1] - 0.001
        if y0 > 0.80:
            break  # below the item table (RegenMed Item section starts ~0.87)
        b = _row_boxes(cols, y0, y1)
        if not _has_ink_overlapping(img1, b["item"]):
            continue
        for key, label in (("lot", "Lot #"), ("exp", "Expiration Date"),
                           ("mfr", "Manufacturer")):
            if not _has_ink_overlapping(img1, b[key]):
                issues.append(_issue("error", f"Lot Log p1 row {j} — {label}",
                                     f"Item is listed but {label} is blank.", b[key], 0))

    # page 1 bottom: RegenMed items need Lot # OR Qty Used
    for j, (y0, y1) in enumerate(t["p1b_rows"]):
        b = _row_boxes(t["p1b_cols"], y0, y1)
        if _has_ink_overlapping(img1, b["item"]) and not (
                _has_ink_overlapping(img1, b["lot"])
                or _has_ink_overlapping(img1, b["qty"])):
            issues.append(_issue("error", f"Lot Log p1 item row {j + 1}",
                                 "Item is listed but both Lot # and Qty Used are blank.",
                                 (b["lot"][0], y0, b["qty"][2], y1), 0))

    if len(pages) < 2:
        issues.append(_issue("warning", "Lot Log",
                             "Only one page was found — the sterilization and packaging "
                             "tables on page 2 were not checked."))
        return issues
    img2 = pages[1]

    def check_ster_table(spec, side):
        for r in range(spec["rows"]):
            y0 = spec["y0"] + r * spec["pitch"] + 0.001
            y1 = y0 + spec["pitch"] - 0.002
            ibox = (spec["item"][0], y0, spec["item"][1], y1)
            lbox = (spec["load"][0], y0, spec["load"][1], y1)
            dbox = (spec["date"][0], y0, spec["date"][1], y1)
            if not _has_ink_overlapping(img2, ibox):
                continue
            voided = _struck_through(img2, (spec["item"][0], y0, spec["item"][1], y1))
            # rule: "Load # or Sterilization Date must not be left blank" --
            # at least one of the two must be present (same OR-logic as the
            # "Lot or Qty Used" rules elsewhere).
            if not (_has_ink_overlapping(img2, lbox)
                    or _has_ink_overlapping(img2, dbox)):
                sev = "warning" if voided else "error"
                msg = ("Item is listed but both Load # and Sterilization Date are blank."
                       + (" Row appears struck-through — confirm it was intentionally voided."
                          if voided else ""))
                issues.append(_issue(sev, f"Lot Log p2 {side} row {r + 1} — Load # / Date",
                                     msg, (lbox[0], y0, dbox[2], y1), 1))

    check_ster_table(t["p2_left"], "left")
    check_ster_table(t["p2_right"], "right")

    # page 2: packaging table (first band is the header)
    pk = t["p2_pack"]
    for r in range(1, pk["rows"]):
        y0 = pk["y0"] + r * pk["pitch"] + 0.001
        y1 = y0 + pk["pitch"] - 0.002
        pbox = (pk["pack"][0], y0, pk["pack"][1], y1)
        lbox = (pk["lot"][0], y0, pk["lot"][1], y1)
        qbox = (pk["qty"][0], y0, pk["qty"][1], y1)
        if _has_ink_overlapping(img2, pbox) and not (
                _has_ink_overlapping(img2, lbox)
                or _has_ink_overlapping(img2, qbox)):
            issues.append(_issue("error", f"Lot Log p2 packaging row {r}",
                                 "Packaging item is listed but both Lot # and Qty Used are blank.",
                                 (lbox[0], y0, qbox[2], y1), 1))
    return issues


CHECKERS = {"MP-F-023": check_mp_f023, "QS-F-049": check_qs_f049,
            "MP-F-021": check_lot_log}


# ------------------------------------------------------------- MP-F-018 (Discard)
def _has_handwriting(img, box, min_h=8):
    """Is there handwriting (not grid lines) in the box?

    Line-aware: ignores horizontal/vertical rule fragments and box-spanning
    lines. Used for table cells where a skewed grid line might cut through.
    """
    H, W = img.shape[:2]
    x0, y0, x1, y1 = (int(box[0] * W), int(box[1] * H),
                      int(box[2] * W), int(box[3] * H))
    c = img[y0:y1, x0:x1]
    if c.size == 0:
        return False
    gray = c if len(c.shape) == 2 else cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    dark = np.where(gray < 200, 255, 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    for i in range(1, n):
        a, hh, ww = (stats[i, cv2.CC_STAT_AREA], stats[i, cv2.CC_STAT_HEIGHT],
                     stats[i, cv2.CC_STAT_WIDTH])
        if a < 15:
            continue  # speckle
        if ww > c.shape[1] * 0.4 and ww / max(hh, 1) > 8:
            continue  # horizontal rule fragment (even if skewed)
        if hh > c.shape[0] * 0.5 and hh / max(ww, 1) > 8:
            continue  # vertical rule fragment
        if hh >= c.shape[0] * 0.85 or ww >= c.shape[1] * 0.85:
            continue  # spans the box: a rule line, not handwriting
        if hh >= min_h:
            return True
    return False


def _ink_blobs(img, search_box):
    """Ink components in search_box as (x0,y0,x1,y1,aspect) fractional.

    Grid-line fragments and speckles are filtered out. Aspect = width/height.
    Dates are wide (aspect > 2); initials/signatures are squarish (aspect ~1).
    """
    H, W = img.shape[:2]
    x0, y0, x1, y1 = (int(search_box[0] * W), int(search_box[1] * H),
                      int(search_box[2] * W), int(search_box[3] * H))
    c = img[y0:y1, x0:x1]
    if c.size == 0:
        return []
    gray = c if len(c.shape) == 2 else cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    dark = np.where(gray < 200, 255, 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    blobs = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < 20:
            continue
        ww, hh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        # filter rule fragments: long thin lines, or lines spanning the box
        if ww > c.shape[1] * 0.4 and ww / max(hh, 1) > 8:
            continue  # horizontal rule
        if hh >= c.shape[0] * 0.85 and ww <= 4:
            continue  # vertical rule (tall and narrow)
        if ww >= c.shape[1] * 0.85 and hh <= 4:
            continue  # horizontal rule (wide and thin)
        bx0 = (stats[i, cv2.CC_STAT_LEFT] + x0) / W
        by0 = (stats[i, cv2.CC_STAT_TOP] + y0) / H
        bx1 = bx0 + ww / W
        by1 = by0 + hh / H
        blobs.append((bx0, by0, bx1, by1, ww / max(hh, 1)))
    return blobs


def _overlaps(blob, box):
    """Does the blob's bbox intersect the box?"""
    return not (blob[2] < box[0] or blob[0] > box[2] or
                blob[3] < box[1] or blob[1] > box[3])


def _has_shape(img, search_box, core_box, kind):
    """Tolerant presence check: is there date-like or initials-like ink
    in core_box?

    The search_box is wider than core_box, so writing slightly outside the
    printed cell still counts. A blob belongs to the zone if its center is
    inside the zone's y-range (prevents initials bleeding into the date
    zone from inflating the measurement). Dates are wide rectangles
    (merged x-span is wide); initials are squarish. Blobs are merged by
    proximity, so a fragmented date ("12 / 10 / 24") still counts as one
    wide shape.
    """
    blobs = []
    for b in _ink_blobs(img, search_box):
        if not _overlaps(b, core_box):
            continue
        cy = (b[1] + b[3]) / 2
        if not (core_box[1] <= cy <= core_box[3]):
            continue  # center not in zone: bleed-over from neighbor
        blobs.append(b)
    if not blobs:
        return False
    if kind == "initials":
        return any(b[4] < 1.8 for b in blobs)
    # date: merged x-span must be wide relative to height
    x0 = min(b[0] for b in blobs)
    x1 = max(b[2] for b in blobs)
    y0 = min(b[1] for b in blobs)
    y1 = max(b[3] for b in blobs)
    H, W = img.shape[:2]
    wpx, hpx = (x1 - x0) * W, (y1 - y0) * H
    return wpx / max(hpx, 1) >= 1.5 and wpx > 0.02 * W


def _expand(box, margin):
    return (box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin)


def _expand_x(box, margin):
    """Expand horizontally only: catches writing spilling into neighbor
    columns without bleeding into adjacent rows."""
    return (box[0] - margin, box[1], box[2] + margin, box[3])


def _has_ink_overlapping(img, core_box, x_margin=0.025):
    """Is there handwriting that overlaps core_box?

    Searches a horizontally-expanded area (writers spill into neighbor
    columns), but counts only ink that actually overlaps the core box.
    Printed grid lines are removed morphologically: handwriting survives,
    straight rules don't.
    """
    H, W = img.shape[:2]
    sx0 = max(0.0, core_box[0] - x_margin)
    sx1 = min(1.0, core_box[2] + x_margin)
    x0, y0, x1, y1 = (int(sx0 * W), int(core_box[1] * H),
                      int(sx1 * W), int(core_box[3] * H))
    c = img[y0:y1, x0:x1]
    if c.size == 0:
        return False
    gray = c if len(c.shape) == 2 else cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    # remove straight grid lines (long thin structures)
    hkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    lines = cv2.morphologyEx(b, cv2.MORPH_OPEN, hkernel)
    lines = cv2.bitwise_or(lines, cv2.morphologyEx(b, cv2.MORPH_OPEN, vkernel))
    ink = cv2.bitwise_and(b, cv2.bitwise_not(lines))
    # does any surviving ink overlap the core box?
    cx0 = int((core_box[0] - sx0) * W)
    cx1 = int((core_box[2] - sx0) * W)
    region = ink[:, max(0, cx0):cx1]
    return bool((region > 0).sum() > 30)


def _detect_status_boxes(img, row_box):
    """Locate the 4 Tissue Status checkbox squares dynamically.

    Absolute calibrated boxes drift when a scan is shifted/rotated relative
    to the calibration sample (a few px is enough to catch a neighbouring
    checkmark). The printed squares are ~square, ~17px at 150dpi, and sit on
    one row; printed letters are narrower. Returns 4 boxes in x-order, or []
    if detection is not clean (caller falls back to calibrated boxes).
    """
    H, W = img.shape[:2]
    y0, y1 = int(row_box[1] * H), int(row_box[3] * H)
    band = img[y0:y1, :]
    _, b = cv2.threshold(band, 180, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sq = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w >= 15 and h >= 15 and w * h <= 600 and 0.8 <= w / h <= 1.25:
            sq.append((x / W, (y + y0) / H, (x + w) / W, (y + h + y0) / H))
    if len(sq) < 4:
        return []
    # the 4 squares share a row: take the 4 with the tightest y-centres
    sq.sort(key=lambda bb: (bb[1] + bb[3]) / 2)
    best = None
    for i in range(len(sq) - 3):
        win = sq[i:i + 4]
        spread = max(bb[3] for bb in win) - min(bb[1] for bb in win)
        if best is None or spread < best[0]:
            best = (spread, win)
    win = sorted(best[1], key=lambda bb: bb[0])
    # sanity: similar sizes, spread across the row, non-overlapping
    ws = [(bb[2] - bb[0]) * W for bb in win]
    if max(ws) > 1.6 * min(ws):
        return []
    if win[3][0] - win[0][0] < 0.3:
        return []
    for a, bb in zip(win, win[1:]):
        if bb[0] < a[2] - 0.005:
            return []
    return win


def _is_checked(img, box):
    """Checkbox / X-box: marked if ink inside (checkmark or X), ignoring the border.

    A checkmark/X is a diagonal stroke; an empty box is hollow. A large inset
    excludes the border; the threshold separates checkmark ink from leakage.
    """
    return ink_fraction(img, box, thresh=170, inset=0.35) > 0.12


def _is_graft_id_listed(img, box):
    """True if the Graft ID cell holds a real ID (not blank, N/A, or a dash)."""
    if not is_filled(img, box):
        return False
    H, W = img.shape[:2]
    x0, y0, x1, y1 = (int(box[0] * W), int(box[1] * H),
                      int(box[2] * W), int(box[3] * H))
    c = img[y0:y1, x0:x1]
    if c.size == 0:
        return False
    gray = c if len(c.shape) == 2 else cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)
    n, _, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    comps = [(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
             for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 20]
    if not comps:
        return False
    # a lone horizontal dash ("—" meaning none) is not an ID
    if len(comps) == 1:
        wpx, hpx = comps[0]
        if wpx > 3 * hpx:
            return False
    text = ocr_text(img, box).strip().upper()
    if not text or NA_RE.match(text) or set(text) <= set("-—_ "):
        return False
    return True


def _find_by_labels(img):
    """Locate printed 'By:' labels in MP-F-018's bottom section via Tesseract.

    Returns a list of (x1, y_center) in page fractions, or [] if OCR fails.
    The signature boxes overlap their printed labels, so a plain is_filled()
    can never report a blank signature; anchoring the value zone just right
    of the 'By:' label excludes the printed ink.
    """
    try:
        from pytesseract import image_to_data, Output
    except ImportError:
        return []
    H, W = img.shape[:2]
    y0, y1 = int(0.70 * H), int(0.88 * H)
    strip = img[y0:y1, :]
    big = cv2.resize(strip, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    try:
        d = image_to_data(big, output_type=Output.DICT, config="--psm 6")
    except Exception:
        return []
    out = []
    n = len(d["text"])
    for i in range(n):
        txt = (d["text"][i] or "").strip()
        if txt not in ("By:", "By"):
            continue
        try:
            conf = float(d["conf"][i])
        except (ValueError, TypeError):
            continue
        if conf < 30:
            continue
        x1 = (d["left"][i] + d["width"][i]) / 2 / W
        yc = (d["top"][i] + d["height"][i] / 2) / 2 / H + y0 / H
        out.append((x1, yc))
    return out


def _sig_value_zone(box, by_labels):
    """Value zone for a 'By' signature box: right of its 'By:' label,
    vertically banded to the label's own line.

    Falls back to the full box when no label is found (e.g. OCR failure).
    The vertical band matters: the box is taller than the signature line
    and its top edge can catch printed text from the row above, whose ink
    alone would mask a blank (erased) signature.
    """
    cands = [(x1, yc) for x1, yc in by_labels
             if box[1] < yc < box[3] and x1 < box[2] - 0.02]
    if not cands:
        return box
    x1, yc = max(cands, key=lambda c: c[0])
    x0 = x1 + 0.008
    if x0 >= box[2] - 0.02:
        return box
    half = 0.022
    y0 = max(box[1] + 0.004, yc - half)
    y1 = min(box[3] - 0.004, yc + half)
    if y1 - y0 < 0.015:
        return (x0, box[1] + 0.004, box[2], box[3] - 0.004)
    return (x0, y0, box[2], y1)


def check_mp_f018(pages):
    """Tissue Discard Form (bonus): top / status / table / bottom rules."""
    t = MP_F_018
    img = pages[0]
    issues = []

    # --- top: Donor #, Authorized By/Date, Reason must not be blank
    # Donor # is checked on the value zone (right of the printed "Donor #:"
    # label); the label's own ink would otherwise mask a blank number.
    if not is_filled(img, t["donor_value_box"]):
        issues.append(_issue("error", "Donor #", "Donor # is blank.", t["donor_value_box"]))

    abox = t["auth_box"]
    if not is_filled(img, abox):
        issues.append(_issue("error", "Discard Authorized By/Date",
                             "Discard Authorized By/Date is blank.", abox))
    else:
        # must have BOTH initials/signature and a date: the date's dashes
        # anchor the search; initials must be a stroke left of the first dash
        dashes = _date_dashes(img, abox)
        if not dashes:
            issues.append(_issue("warning", "Discard Authorized By/Date",
                                 "Could not verify both initials and date are present — "
                                 "please confirm visually.", abox))
        else:
            ibox = (abox[0], abox[1], min(dashes) - 0.015, abox[3])
            if not _has_initials(img, ibox):
                issues.append(_issue("error", "Discard Authorized By",
                                     "Date is present but the authorizing initials/signature "
                                     "are missing.", ibox))

    if not is_filled(img, t["reason_box"]):
        issues.append(_issue("error", "Reason for Discard",
                             "Reason for Discard is blank.", t["reason_box"]))

    # --- Tissue Status: exactly one box checked (squares located dynamically
    # so a shifted/rotated scan can't misalign the calibrated boxes)
    status_labels = ["Unprocessed Tissue", "In Processing Tissue",
                     "Unreleased Packaged Tissue", "Released Packaged Tissue"]
    dyn_boxes = _detect_status_boxes(img, t["status_row_box"])
    if dyn_boxes:
        status_items = list(zip(status_labels, dyn_boxes))
    else:
        status_items = list(t["status_boxes"].items())
    checked = [name for name, box in status_items if _is_checked(img, box)]
    if len(checked) == 0:
        issues.append(_issue("error", "Tissue Status",
                             "No Tissue Status box is checked.", t["status_row_box"]))
    elif len(checked) > 1:
        issues.append(_issue("error", "Tissue Status",
                             f"Multiple Tissue Status boxes are checked: "
                             f"{', '.join(checked)}.", t["status_row_box"]))

    # --- middle table: X for each listed tissue; Graft ID vs Status linkage
    has_real_graft_id = False
    cols, y0_0, pitch = t["table_cols"], t["table_y0"], t["table_pitch"]
    for r in range(t["table_rows"]):
        y0, y1 = y0_0 + r * pitch, y0_0 + (r + 1) * pitch - 0.002
        desc_box = (cols["desc"][0] + 0.005, y0 + 0.003, cols["desc"][1] - 0.005, y1)
        if not _has_handwriting(img, desc_box):
            continue  # empty row
        x_box = (cols["x"][0], y0 + 0.002, cols["x"][1], y1)
        if not _is_checked(img, x_box):
            issues.append(_issue("error", f"Tissue row {r + 1} — X",
                                 "Tissue is listed but the X box is not marked.", x_box))
        graft_box = (cols["graft"][0] + 0.005, y0 + 0.003, cols["graft"][1] - 0.005, y1)
        if _is_graft_id_listed(img, graft_box):
            has_real_graft_id = True

    if len(checked) == 1:
        status = checked[0]
        if has_real_graft_id and status not in (
                "Unreleased Packaged Tissue", "Released Packaged Tissue"):
            issues.append(_issue(
                "error", "Tissue Status",
                f"A Graft ID is listed but Tissue Status is '{status}' — with a "
                f"Graft ID it must be Unreleased or Released Packaged Tissue.",
                t["status_boxes"][status]))
        if not has_real_graft_id and status not in (
                "Unprocessed Tissue", "In Processing Tissue"):
            issues.append(_issue(
                "error", "Tissue Status",
                f"No Graft ID is listed but Tissue Status is '{status}' — without "
                f"a Graft ID it must be Unprocessed or In Processing Tissue.",
                t["status_boxes"][status]))

    # --- bottom: none of the fields may be blank; dates get format check.
    # Signature boxes overlap their printed labels, so the value zone is
    # anchored just right of each 'By:' label (found via one OCR pass).
    by_labels = _find_by_labels(img)
    bottom = [
        ("Tissue Discarded By", t["discarded_by_box"], False, True),
        ("Confirmed By", t["confirmed_by_box"], False, True),
        ("Discard Date", t["discard_date_box"], True, False),
        ("Released Packaged — FreezerPro Updated By", t["released_by_box"], False, True),
        ("Released Packaged — Date", t["released_date_box"], True, False),
        ("Donor Chart — Log / FreezerPro Updated By", t["donorchart_by_box"], False, True),
        ("Donor Chart — Date", t["donorchart_date_box"], True, False),
    ]
    for field, box, is_date, is_sig in bottom:
        vbox = _sig_value_zone(box, by_labels) if is_sig else box
        if not is_filled(img, vbox):
            issues.append(_issue("error", field, f"{field} is blank.", vbox))
        elif is_date:
            issues.extend(check_date_zone(img, box, field))
    return issues


CHECKERS["MP-F-018"] = check_mp_f018
