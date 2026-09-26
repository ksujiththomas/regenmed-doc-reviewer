"""Form templates: calibrated fractional coordinates for each supported form.

Coordinates are fractions of page width/height, calibrated from the reference
scans. Generous insets are applied at check time to absorb scan shift/scale.
"""

# ---------------------------------------------------------------- MP-F-023
MP_F_023 = {
    "code": "MP-F-023",
    "name": "Tissue Processing Form",
    "footer_markers": ["MP-F-023"],
    "title_markers": ["MS Processing Instructions", "Tissue Open Checklist"],
    "header_cols": {  # label -> (x0, x1) ; value row y 0.12-0.15 unless noted
        "Donor #": (0.04, 0.16),
        "Cross Reference": (0.16, 0.28),
        "Sex": (0.28, 0.34),
        "Age": (0.34, 0.40),
        "Date of Recovery": (0.40, 0.52),
        "Instruction Verification": (0.52, 0.68),
        "Date of Processing": (0.68, 0.78),
        "Clean Room Log": (0.78, 0.88),
        "Tissue Checked In": (0.88, 0.97),
    },
    "header_value_y": (0.12, 0.15),
    # Donor # value is handwritten right of the printed label; the value zone
    # excludes the printed "Donor #" text (which would otherwise make the
    # blank check vacuous).
    "donor_value_box": (0.095, 0.078, 0.16, 0.102),
    # "Verified By" cell under Donor #: initials required; zone excludes the
    # printed label at left.
    "verified_by_box": (0.10, 0.118, 0.16, 0.148),
    # stacked By/Date cells: (top, mid, bottom). Writers often sign high,
    # overlapping the printed label (which ends ~y 0.112), so the initials
    # zone starts at 0.115. Mid is 0.130 to keep the date's digit tops out
    # of the initials zone.
    "by_date_split": (0.115, 0.130, 0.15),
    # writers also spill across the column divider (e.g. "MW" written over the
    # Clean Room Log | Tissue Checked In boundary), so each cell's initials
    # zone extends slightly into its neighbour: (cell_x0, cell_x1, ibox_x0, ibox_x1)
    "stacked_cells": {
        # (cell_x0, cell_x1, initials_x0, initials_x1); initials zone is the
        # upper part of the cell (by_date_split). Clean Room Log's initials
        # zone is capped at 0.855 so the Tissue Checked In "MW" spilling
        # left across the divider isn't mistaken for its own initials.
        "Clean Room Log": (0.78, 0.88, 0.775, 0.855),
        "Tissue Checked In": (0.88, 0.97, 0.855, 0.97),
    },
    "date_fields": ["Date of Recovery", "Date of Processing"],
    "ops_review_box": (0.54, 0.455, 0.92, 0.472),
    "tissue_cols": {
        "name": (0.04, 0.32),
        "produced": (0.475, 0.578),
        "packaged": (0.578, 0.681),
    },
    "produced_inset": (0.485, 0.568),
    "packaged_inset": (0.588, 0.671),
    # Fallback FRZ/FD row centers if dynamic detection fails (fractional y).
    "frfd_fallback_centers": [
        0.4970, 0.5134, 0.5291, 0.5450, 0.5611, 0.5934, 0.6257, 0.6411, 0.6736,
        0.7059, 0.7220, 0.7380, 0.7539, 0.7859, 0.8020, 0.8177, 0.8339,
    ],
    # Pre-printed tissue names, top to bottom (for issue labels).
    "tissue_names": [
        "Posterior Tibialis", "Anterior Tibialis", "Peroneus Longus", "Gracilis",
        "Semitendinosus", "Patellar Ligament", "Femoral Head", "Humeral Head",
        "Tri-Cortical Block", "Cancellous 1-10 mm", "Cancellous 4-10 mm",
        "Cancellous 1-4 mm", "Cancellous 3-6 mm", "Tibia Shaft", "Humerus Shaft",
        "Femur Shaft", "Fibula Shaft",
    ],
}

# ---------------------------------------------------------------- QS-F-049
QS_F_049 = {
    "code": "QS-F-049",
    "name": "Processing Room Cleaning Log",
    "footer_markers": ["QS-F-049"],
    "title_markers": ["Technical/Quality Review", "Disposition Statement"],
    "tech_x": (0.775, 0.868),
    "qual_x": (0.868, 0.960),
    "row_bands": [
        (0.183, 0.243), (0.243, 0.273), (0.273, 0.313), (0.313, 0.350),
        (0.350, 0.380), (0.380, 0.423), (0.423, 0.463), (0.463, 0.493),
        (0.493, 0.528), (0.528, 0.590),
    ],
    "initials_frac": 0.45,   # top fraction of cell = initials zone
    "inc_box": (0.185, 0.55, 0.35, 0.575),
    "status_box": (0.44, 0.55, 0.66, 0.585),
}

# ---------------------------------------------------------------- Lot Log (MP-F-021)
LOT_LOG = {
    "code": "MP-F-021",
    "name": "MS Processing & Packaging Lot Log",
    "footer_markers": ["MP-F-021"],
    "title_markers": ["Lot Log", "MS Processing & Packaging"],
    # page 1: item table
    "p1_cols": {"item": (0.0447, 0.2588), "lot": (0.2588, 0.4941),
                "exp": (0.4941, 0.6859), "mfr": (0.6859, 0.9441)},
    "p1_hlines": [0.3036, 0.3255, 0.3477, 0.3723, 0.39, 0.4077, 0.425, 0.4427,
                  0.46, 0.4777, 0.4955, 0.5127, 0.5305, 0.5482, 0.5655, 0.5832,
                  0.6005, 0.6182, 0.6355, 0.6532, 0.6709, 0.6886, 0.7059, 0.7236,
                  0.7414, 0.7591, 0.7764, 0.7941, 0.8118, 0.8295, 0.8468],
    # page 1: bottom RegenMed item table
    "p1b_rows": [(0.872, 0.890), (0.890, 0.908), (0.908, 0.926), (0.926, 0.944)],
    "p1b_cols": {"item": (0.0447, 0.3765), "lot": (0.3765, 0.6859),
                 "qty": (0.6859, 0.9447)},
    # page 2: left sterilization table (34 rows)
    "p2_left": {"rows": 34, "y0": 0.09, "pitch": 0.0176,
                "item": (0.045, 0.251), "load": (0.251, 0.332), "date": (0.332, 0.473)},
    # page 2: right sterilization table (30 rows)
    "p2_right": {"rows": 30, "y0": 0.09, "pitch": 0.0176,
                 "item": (0.495, 0.723), "load": (0.723, 0.804), "date": (0.804, 0.942)},
    # page 2: packaging table
    "p2_pack": {"y0": 0.7064, "pitch": 0.0177, "rows": 11,
                "pack": (0.4947, 0.6294), "lot": (0.6294, 0.8041), "qty": (0.8041, 0.9406)},
}

# ---------------------------------------------------------------- MP-F-018 (Discard)
MP_F_018 = {
    "code": "MP-F-018",
    "name": "Tissue Discard Form",
    "footer_markers": ["MP-F-018"],
    "title_markers": ["Tissue Discard Form", "DISCARD FORM"],
    # top fields (value zones exclude printed labels)
    "donor_box": (0.075, 0.100, 0.180, 0.130),
    "auth_box": (0.560, 0.098, 0.965, 0.138),
    "reason_box": (0.200, 0.130, 0.965, 0.158),
    # Tissue Status checkboxes (17px at 150dpi; tight boxes avoid label text)
    "status_boxes": {
        "Unprocessed Tissue": (0.152, 0.171, 0.171, 0.187),
        "In Processing Tissue": (0.322, 0.176, 0.341, 0.192),
        "Unreleased Packaged Tissue": (0.495, 0.181, 0.514, 0.197),
        "Released Packaged Tissue": (0.726, 0.187, 0.744, 0.203),
    },
    "status_row_box": (0.060, 0.160, 0.965, 0.198),
    # middle table: Graft IDs | Tissue Description | Storage Location | X
    "table_cols": {
        "graft": (0.035, 0.185),
        "desc": (0.185, 0.735),
        "storage": (0.735, 0.925),
        "x": (0.925, 0.968),
    },
    "table_y0": 0.2485,   # first data row top (below header)
    "table_pitch": 0.0270,
    "table_rows": 17,
    # bottom fields (y calibrated from deskewed 150dpi scan)
    "discarded_by_box": (0.175, 0.733, 0.345, 0.773),
    "confirmed_by_box": (0.470, 0.733, 0.615, 0.773),
    "discard_date_box": (0.770, 0.733, 0.960, 0.773),
    "released_by_box": (0.415, 0.773, 0.615, 0.806),
    "released_date_box": (0.770, 0.773, 0.960, 0.806),
    "donorchart_by_box": (0.415, 0.813, 0.615, 0.862),
    "donorchart_date_box": (0.770, 0.813, 0.960, 0.862),
}

FORMS = {"MP-F-023": MP_F_023, "QS-F-049": QS_F_049, "MP-F-021": LOT_LOG,
         "MP-F-018": MP_F_018}
