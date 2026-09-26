"""Per-form validation rule engines. Each returns a list of issue dicts:

    {"severity": "error"|"warning", "field": str, "message": str,
     "box": (x0,y0,x1,y1) | None, "page": int}
"""
import re
import cv2
import numpy as np

from .vision import (is_filled, ink_fraction, count_x_clusters, count_y_clusters,
                     has_diagonal_slash, ocr_text)
from .forms import MP_F_023, QS_F_049, LOT_LOG

DATE_RE = re.compile(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2}(\d{2})?$")
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


def check_date_zone(img, box, field, page=0):
    """Date must be present and in MM/DD/YY (or MM-DD-YY) shape."""
    if not is_filled(img, box):
        return [_issue("error", field, "Date is blank.", box, page)]
    text = ocr_text(img, box, "--psm 7 -c tessedit_char_whitelist=0123456789/-")
    text = text.replace(" ", "")
    if DATE_RE.match(text):
        return []
    # structural fallback: too many strokes => probably words, not a date
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in (box[0] * W, box[1] * H, box[2] * W, box[3] * H)]
    c = img[y0:y1, x0:x1]
    _, b = cv2.threshold(c, 175, 255, cv2.THRESH_BINARY_INV)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    comps = sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 25)
    if comps > 12:
        return [_issue("error", field,
                       "Date format looks invalid (expected MM/DD/YY).", box, page)]
    return [_issue("warning", field,
                   "Date is present but the MM/DD/YY format could not be "
                   "machine-verified — please confirm visually.", box, page)]


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
            if not is_filled(img, ibox):
                issues.append(_issue("error", f"{label} — By (initials)",
                                     "Initials are missing.", ibox))
            issues.extend(check_date_zone(img, dbox, f"{label} — Date"))
        elif label == "Donor #":
            vbox = (cx0, y0, cx1, y1)
            if not (is_filled(img, t["donor_label_box"]) or is_filled(img, vbox)):
                issues.append(_issue("error", "Donor #", "Donor # is blank.", vbox))
        else:
            if not is_filled(img, box):
                issues.append(_issue("error", label, f"{label} is blank.", box))
            elif label in t["date_fields"]:
                issues.extend(check_date_zone(img, box, label))

    # Operations Manager Review: needs initials AND date.
    # The two are written side-by-side; require ink spanning a wide x-range
    # with at least two separated clusters.
    obox = t["ops_review_box"]
    n = count_x_clusters(img, obox)
    span = _ink_x_span(img, obox)
    if n == 0:
        issues.append(_issue("error", "Operations Manager Review",
                             "Review initials/date are missing.", obox))
    elif n < 2 or span < t["ops_review_min_span"]:
        issues.append(_issue("warning", "Operations Manager Review",
                             "Only one entry found — initials and date are both required.",
                             obox))

    # Tissue rows: named tissue must have # Produced and/or # Packaged
    ncol = t["tissue_cols"]
    for i, (ry0, ry1) in enumerate(t["tissue_rows"]):
        nbox = (ncol["name"][0], ry0, ncol["name"][1], ry1)
        pbox = (t["produced_inset"][0], ry0, t["produced_inset"][1], ry1)
        kbox = (t["packaged_inset"][0], ry0, t["packaged_inset"][1], ry1)
        name = is_filled(img, nbox, inset=0.08)
        prod = is_filled(img, pbox, inset=0.05)
        pack = is_filled(img, kbox, inset=0.05)
        label = f"Tissue row {i + 1}"
        if name and not (prod or pack):
            issues.append(_issue("error", label,
                                 "Tissue is listed but # Produced and # Packaged are both blank.",
                                 (ncol["produced"][0], ry0, ncol["packaged"][1], ry1)))
        elif (prod or pack) and not name:
            issues.append(_issue("error", label,
                                 "Quantity entered but the tissue name is blank.", nbox))
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
            ini = is_filled(img, ibox)
            dat = is_filled(img, dbox)
            if ini and dat:
                issues.extend(check_date_zone(img, dbox, f"{field} date"))
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

    # page 1: item table (skip header band)
    cols = t["p1_cols"]
    for j in range(1, len(t["p1_hlines"]) - 1):
        y0, y1 = t["p1_hlines"][j] + 0.001, t["p1_hlines"][j + 1] - 0.001
        b = _row_boxes(cols, y0, y1)
        if not is_filled(img1, b["item"]):
            continue
        for key, label in (("lot", "Lot #"), ("exp", "Expiration Date"),
                           ("mfr", "Manufacturer")):
            if not is_filled(img1, b[key]):
                issues.append(_issue("error", f"Lot Log p1 row {j} — {label}",
                                     f"Item is listed but {label} is blank.", b[key], 0))

    # page 1 bottom: RegenMed items need Lot # OR Qty Used
    for j, (y0, y1) in enumerate(t["p1b_rows"]):
        b = _row_boxes(t["p1b_cols"], y0, y1)
        if is_filled(img1, b["item"]) and not (
                is_filled(img1, b["lot"]) or is_filled(img1, b["qty"])):
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
            if not is_filled(img2, ibox):
                continue
            voided = _struck_through(img2, (spec["item"][0], y0, spec["item"][1], y1))
            for box, label in ((lbox, "Load #"), (dbox, "Sterilization Date")):
                if not is_filled(img2, box):
                    sev = "warning" if voided else "error"
                    msg = (f"Item is listed but {label} is blank."
                           + (" Row appears struck-through — confirm it was intentionally voided."
                              if voided else ""))
                    issues.append(_issue(sev, f"Lot Log p2 {side} row {r + 1} — {label}",
                                         msg, box, 1))

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
        if is_filled(img2, pbox) and not (is_filled(img2, lbox) or is_filled(img2, qbox)):
            issues.append(_issue("error", f"Lot Log p2 packaging row {r}",
                                 "Packaging item is listed but both Lot # and Qty Used are blank.",
                                 (lbox[0], y0, qbox[2], y1), 1))
    return issues


CHECKERS = {"MP-F-023": check_mp_f023, "QS-F-049": check_qs_f049, "MP-F-021": check_lot_log}
