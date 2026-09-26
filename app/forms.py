"""Form templates: calibrated fractional coordinates for each supported form.

Coordinates are fractions of page width/height, calibrated from the reference
scans. Generous insets are applied at check time to absorb scan shift/scale.
"""

# ---------------------------------------------------------------- MP-F-023
MP_F_023 = {
    "code": "MP-F-023",
    "name": "MS Processing Instructions / Tissue Open Checklist",
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
    "donor_label_box": (0.04, 0.08, 0.16, 0.10),  # Donor # lives in label row
    "by_date_split": (0.12, 0.133, 0.15),  # (top, mid, bottom) for stacked By/Date
    "date_fields": ["Date of Recovery", "Date of Processing"],
    "ops_review_box": (0.54, 0.455, 0.92, 0.472),
    "ops_review_min_span": 0.09,  # ink x-span fraction needed for initials+date
    "tissue_cols": {
        "name": (0.04, 0.32),
        "produced": (0.50, 0.62),
        "packaged": (0.62, 0.72),
    },
    "produced_inset": (0.51, 0.61),
    "packaged_inset": (0.63, 0.71),
    "tissue_rows": [
        (0.510, 0.530), (0.530, 0.545), (0.545, 0.560), (0.560, 0.575),
        (0.575, 0.595), (0.605, 0.625), (0.635, 0.650), (0.650, 0.665),
        (0.675, 0.695), (0.705, 0.720), (0.720, 0.735), (0.735, 0.750),
        (0.750, 0.770), (0.780, 0.795), (0.795, 0.810), (0.810, 0.825),
        (0.825, 0.845),
    ],
}

# ---------------------------------------------------------------- QS-F-049
QS_F_049 = {
    "code": "QS-F-049",
    "name": "Technical/Quality Review and Disposition Statement",
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

FORMS = {"MP-F-023": MP_F_023, "QS-F-049": QS_F_049, "MP-F-021": LOT_LOG}
