"""
LiveDoc — Full Image-to-PDF Pipeline Test
Tests real scanned document images through the entire pipeline and reports findings.
"""

import sys
import os
import cv2
import numpy as np
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.document_detector import DocumentDetector
from src.layout_analyzer import LayoutAnalyzer
from src.ocr_engine import OCREngine
from src.pdf_exporter import PDFExporter

TEST_IMAGES = [
    ("Theory of Computation (book page with tables)", r"tested doc by user export\test 1\WhatsApp Image 2026-10-31 at 18.37.10.jpeg"),
    ("English comprehension (dense paragraph page)", r"tested doc by user export\test 1\engtest.jpeg"),
    ("Book cover / logo / image heavy page",          r"tested doc by user export\test 1\lastpagebook.jpeg"),
    ("Two-page spread (facing page bleed risk)",       r"tested doc by user export\test 1\page 2.jpeg"),
]

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

os.makedirs("test_pipeline_outputs", exist_ok=True)

detector = DocumentDetector()
layout  = LayoutAnalyzer()
print("Initializing OCR Engine (PaddleOCR)...")
ocr_engine = OCREngine()
exporter = PDFExporter()

findings = []

def assess(label, condition, detail="", warn_only=False):
    sym = PASS if condition else (WARN if warn_only else FAIL)
    findings.append((label, condition, detail, warn_only))
    print(f"  {sym}  {label}")
    if detail:
        print(f"         {detail}")

print("\n" + "="*65)
print("  LiveDoc — Real Image Pipeline Test")
print("="*65)

for img_label, img_path in TEST_IMAGES:
    if not os.path.exists(img_path):
        print(f"\n[SKIP] Image not found: {img_path}")
        continue

    print(f"\n{'─'*65}")
    print(f"  IMAGE: {img_label}")
    print(f"  PATH : {img_path}")
    print(f"{'─'*65}")

    frame = cv2.imread(img_path)
    if frame is None:
        assess(f"[{img_label}] Load image", False, f"cv2.imread returned None for {img_path}")
        continue
    assess(f"[{img_label}] Load image", True, f"Shape: {frame.shape}")

    # ── 1. Quality Assessment ──────────────────────────────────────────
    t0 = time.time()
    quality = detector.assess_frame_quality(frame)
    q_status = quality["status"]
    q_msg    = quality["message"]
    q_score  = quality.get("score", 0)
    assess(f"[{img_label}] Quality assessment", q_status not in ("BLURRY",), 
           f"Status={q_status} | Score={q_score:.1f} | {q_msg}", warn_only=(q_status == "BLURRY"))

    # ── 2. Document Detection & Flattening ────────────────────────────
    corners = detector.find_document(frame)
    has_corners = corners is not None
    assess(f"[{img_label}] Document corner detection", has_corners,
           f"Corners found: {corners.tolist() if has_corners else 'None — using full frame fallback'}", warn_only=not has_corners)

    if has_corners:
        flat = detector.flatten_document(frame, corners)
    else:
        flat = frame.copy()
    flat_preprocessed = detector.preprocess_for_ocr(flat)
    assess(f"[{img_label}] Flatten + preprocess", flat_preprocessed is not None,
           f"Output size: {flat_preprocessed.shape}")

    # ── 3. Layout Analysis ────────────────────────────────────────────
    layout_boxes = layout.analyze(flat_preprocessed)
    types_found = list(set(b["type"] for b in layout_boxes))
    has_tables = any(b["type"] == "Table" for b in layout_boxes)
    has_images = any(b["type"] == "Image" for b in layout_boxes)
    assess(f"[{img_label}] Layout analysis", len(layout_boxes) > 0,
           f"Boxes found: {len(layout_boxes)} | Types: {types_found}")
    assess(f"[{img_label}] Table detection", has_tables,
           f"{'Tables detected by CV morph' if has_tables else 'No table borders found by layout analyzer — OCR token-grid recovery will try'}",
           warn_only=not has_tables)
    assess(f"[{img_label}] Image/Figure detection", has_images,
           f"{'Graphic regions detected' if has_images else 'No graphic regions flagged'}",
           warn_only=not has_images)

    # ── 4. OCR + Structured Extraction ───────────────────────────────
    t_ocr = time.time()
    result = ocr_engine.extract_structured_document(flat_preprocessed, layout_boxes)
    ocr_ms = (time.time() - t_ocr) * 1000

    elements  = result.get("elements", [])
    markdown  = result.get("markdown", "")
    title     = result.get("title", "")
    validation= result.get("validation", {})

    word_count = len(markdown.split())
    has_text   = word_count > 10
    assess(f"[{img_label}] OCR text extraction", has_text,
           f"Words extracted: {word_count} | OCR time: {ocr_ms:.0f}ms")

    assess(f"[{img_label}] Title detection", bool(title and title != "Untitled Document"),
           f"Title: '{title}'", warn_only=True)

    el_types = [e["type"] for e in elements]
    has_table_el = "Table" in el_types
    has_image_el = "Image" in el_types
    assess(f"[{img_label}] Table element in output", has_table_el,
           f"Elements: {el_types}", warn_only=not has_table_el)
    assess(f"[{img_label}] Image element in output", has_image_el,
           f"{'Image crop saved to documents/figures/' if has_image_el else 'No image element extracted'}",
           warn_only=not has_image_el)

    # ── 5. Confidence Score ───────────────────────────────────────────
    conf = validation.get("confidence", {})
    conf_score = conf.get("score", 0)
    conf_grade = conf.get("grade", "N/A")
    needs_esc  = conf.get("needs_escalation", False)
    assess(f"[{img_label}] Confidence score", conf_score >= 70,
           f"Score={conf_score}% | Grade={conf_grade} | NeedsAI={needs_esc}", warn_only=conf_score < 70)

    # ── 6. Facing-page bleed detection check ─────────────────────────
    # Look for stray margin words that shouldn't appear in output
    bleed_ok = True
    for el in elements:
        txt = el.get("content", "") or el.get("text", "")
        if len(txt.split()) <= 2 and el.get("type") in ("Paragraph", "Heading"):
            bleed_ok = False
            break
    assess(f"[{img_label}] Facing-page bleed suppression", bleed_ok,
           f"{'No stray 1-2 word snippets in output' if bleed_ok else 'Some 1-2 word snippets found — possible bleed-through or noise'}",
           warn_only=not bleed_ok)

    # ── 7. PDF Export ─────────────────────────────────────────────────
    safe_name = img_label.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace(",", "")[:40]
    pdf_path = f"test_pipeline_outputs/{safe_name}.pdf"
    try:
        exporter.create_digital_reconstructed_pdf(elements, pdf_path)
        pdf_exists = os.path.isfile(pdf_path) and os.path.getsize(pdf_path) > 500
        assess(f"[{img_label}] PDF export", pdf_exists,
               f"PDF saved → {pdf_path} ({os.path.getsize(pdf_path)//1024}KB)")
    except Exception as e:
        assess(f"[{img_label}] PDF export", False, f"Exception: {e}")

    # ── 8. Print sample extracted text ───────────────────────────────
    print(f"\n  {INFO}  Extracted Markdown Preview (first 600 chars):")
    print("  " + "·"*60)
    preview = markdown[:600].replace("\n", "\n  ")
    print(f"  {preview}")
    if len(markdown) > 600:
        print(f"  ... (+{len(markdown)-600} more chars)")
    print("  " + "·"*60)

    # ── 9. Element breakdown ─────────────────────────────────────────
    print(f"\n  {INFO}  Structured Elements Breakdown:")
    for i, el in enumerate(elements[:12], 1):
        t  = el.get("type", "?")
        c  = (el.get("content", "") or el.get("text", ""))[:70]
        cf = el.get("confidence", 1.0)
        print(f"    [{i:02d}] {t:<12} conf={cf:.2f}  \"{c}\"")
    if len(elements) > 12:
        print(f"    ... and {len(elements)-12} more elements")


# ── FINAL SUMMARY ────────────────────────────────────────────────────
print("\n" + "="*65)
print("  FLAW ANALYSIS SUMMARY")
print("="*65)

passed = sum(1 for _, ok, _, w in findings if ok)
warned = sum(1 for _, ok, _, w in findings if not ok and w)
failed = sum(1 for _, ok, _, w in findings if not ok and not w)
total  = len(findings)

print(f"  Total checks : {total}")
print(f"  Passed       : {passed}")
print(f"  Warnings     : {warned}  (non-critical)")
print(f"  Failed       : {failed}")

if warned or failed:
    print(f"\n  Issues Found:")
    for label, ok, detail, w in findings:
        if not ok:
            sym = WARN if w else FAIL
            print(f"  {sym}  {label}")
            if detail:
                print(f"         → {detail}")

print("="*65)
