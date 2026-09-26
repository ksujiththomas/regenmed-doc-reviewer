---
title: RegenMed Document Reviewer
sdk: docker
app_port: 7860
---

# RegenMed Document Reviewer

Upload a filled processing-form PDF — the app detects the form type and flags
missing or inconsistent fields per the RegenMed hackathon "Internal Document
Reviewer" challenge.

**Supported forms**
- **MP-F-023** — Tissue Processing Form
- **QS-F-049** — Processing Room Cleaning Log
- **MP-F-021** — MS Processing & Packaging Lot Log

**How it works**
1. The PDF is rendered and deskewed (OpenCV).
2. The form type is identified from footer/title markers (Tesseract OCR).
3. Per-form rule engines check every required field using ink-presence
   analysis, handwriting-structure checks, N/A slash detection, and
   strike-through (voided row) detection.
4. Results show an errors/warnings list plus annotated page images
   (red = error, orange = warning).

Fully local pipeline — no external APIs, no data leaves the server.
