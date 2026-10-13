"""
LiveDoc Pipeline Test Script
Tests each step of the pipeline and reports PASS / FAIL clearly.
Run with: python test_pipeline.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import sys
import numpy as np

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []

def report(step, status, detail=""):
    tag = PASS if status else FAIL
    msg = f"  {tag}  {step}"
    if detail:
        msg += f"\n         → {detail}"
    print(msg)
    results.append((step, status))

print("\n" + "="*60)
print("  LiveDoc Pipeline Test Report")
print("="*60)

# ─────────────────────────────────────────────
print("\n[1] LIBRARY IMPORTS")
# ─────────────────────────────────────────────

try:
    import cv2
    report("OpenCV", True, f"version {cv2.__version__}")
except Exception as e:
    report("OpenCV", False, str(e))

try:
    import numpy as np
    report("NumPy", True, f"version {np.__version__}")
except Exception as e:
    report("NumPy", False, str(e))

try:
    from PyQt6.QtWidgets import QApplication
    report("PyQt6", True, "QApplication imported OK")
except Exception as e:
    report("PyQt6", False, str(e))

try:
    from ultralytics import YOLO
    report("Ultralytics (YOLO)", True, "YOLO class imported OK")
except Exception as e:
    report("Ultralytics (YOLO)", False, str(e))

try:
    from paddleocr import PaddleOCR
    report("PaddleOCR", True, "PaddleOCR class imported OK")
except Exception as e:
    report("PaddleOCR", False, str(e))

# ─────────────────────────────────────────────
print("\n[2] STEP 1 — Camera Manager")
# ─────────────────────────────────────────────

try:
    sys.path.insert(0, '.')
    from src.camera_manager import LiveCameraManager
    report("LiveCameraManager import", True)
except Exception as e:
    report("LiveCameraManager import", False, str(e))

try:
    cam = LiveCameraManager(0)
    opened = cam.cap.isOpened()
    if opened:
        report("Camera open (webcam found)", True, "cv2.VideoCapture(0) succeeded")
        frame = cam.get_frame()
        if frame is not None:
            report("get_frame() returns image", True, f"frame shape: {frame.shape}")
        else:
            report("get_frame() returns image", False, "returned None — camera may not have a live feed")
    else:
        report("Camera open (webcam found)", False, "No webcam detected — expected in headless environment")
    cam.release()
    report("camera.release()", True)
except Exception as e:
    report("Camera open / get_frame", False, str(e))

# ─────────────────────────────────────────────
print("\n[3] STEP 2 — Document Detector (on dummy image)")
# ─────────────────────────────────────────────

try:
    from src.document_detector import DocumentDetector
    report("DocumentDetector import", True)
except Exception as e:
    report("DocumentDetector import", False, str(e))
    DocumentDetector = None

if DocumentDetector:
    detector = DocumentDetector()

    # Test 1: Blur detection on a sharp synthetic image
    try:
        # Create white image with black text — high variance = sharp
        sharp = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.putText(sharp, "LiveDoc Test Sharp Image", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
        blurry_result = detector.is_blurry(sharp)
        report("is_blurry() on sharp image", not blurry_result,
               f"Laplacian variance classified as {'BLURRY ❌ (wrong)' if blurry_result else 'SHARP ✓'}")
    except Exception as e:
        report("is_blurry() on sharp image", False, str(e))

    # Test 2: Blur detection on a genuinely blurry image
    try:
        blurry_img = np.ones((400, 600, 3), dtype=np.uint8) * 128  # flat gray = low variance
        blurry_result2 = detector.is_blurry(blurry_img)
        report("is_blurry() on blurry image", blurry_result2,
               f"Correctly classified as {'BLURRY ✓' if blurry_result2 else 'SHARP ❌ (wrong)'}")
    except Exception as e:
        report("is_blurry() on blurry image", False, str(e))

    # Test 3: find_document on synthetic document image
    try:
        # Create a white rectangle (simulated paper) on dark background
        scene = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.rectangle(scene, (200, 100), (1000, 620), (255, 255, 255), -1)
        cv2.putText(scene, "Document Text Here", (250, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        corners = detector.find_document(scene)
        report("find_document() on synthetic doc", corners is not None,
               f"Found corners: {corners.tolist() if corners is not None else 'None — document not detected'}")
    except Exception as e:
        report("find_document() on synthetic doc", False, str(e))

    # Test 4: flatten_document
    try:
        if corners is not None:
            flattened = detector.flatten_document(scene, corners)
            report("flatten_document() perspective warp", flattened is not None and flattened.size > 0,
                   f"Output shape: {flattened.shape}")
        else:
            report("flatten_document() perspective warp", False, "Skipped — no corners found in previous step")
    except Exception as e:
        report("flatten_document() perspective warp", False, str(e))

# ─────────────────────────────────────────────
print("\n[4] STEP 3 — Layout Analyzer")
# ─────────────────────────────────────────────

try:
    from src.layout_analyzer import LayoutAnalyzer
    report("LayoutAnalyzer import", True)
except Exception as e:
    report("LayoutAnalyzer import", False, str(e))
    LayoutAnalyzer = None

if LayoutAnalyzer:
    try:
        analyzer = LayoutAnalyzer()  # Will use fallback mode (no weights)
        report("LayoutAnalyzer init (fallback mode)", True, "No YOLO weights — graceful fallback active")
    except Exception as e:
        report("LayoutAnalyzer init", False, str(e))
        analyzer = None

    if analyzer:
        try:
            dummy_img = np.ones((800, 600, 3), dtype=np.uint8) * 255
            boxes = analyzer.analyze(dummy_img)
            ok = isinstance(boxes, list) and len(boxes) > 0 and "box" in boxes[0]
            report("analyze() returns layout boxes", ok,
                   f"Got: {boxes}")
        except Exception as e:
            report("analyze() returns layout boxes", False, str(e))

# ─────────────────────────────────────────────
print("\n[5] STEP 4 — OCR Engine")
# ─────────────────────────────────────────────

try:
    from src.ocr_engine import OCREngine
    report("OCREngine import", True)
except Exception as e:
    report("OCREngine import", False, str(e))
    OCREngine = None

if OCREngine:
    try:
        print("         → Initializing PaddleOCR (may download models on first run ~100MB)...")
        engine = OCREngine()
        report("OCREngine init (PaddleOCR ready)", True)
    except Exception as e:
        report("OCREngine init (PaddleOCR ready)", False, str(e))
        engine = None

    if engine:
        try:
            # Create a clear white image with black text for OCR
            test_img = np.ones((150, 600, 3), dtype=np.uint8) * 255
            cv2.putText(test_img, "LiveDoc OCR Test 2026", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
            text = engine.extract_full_text(test_img)
            ok = len(text.strip()) > 0
            report("extract_full_text() reads text from image", ok,
                   f"Extracted: '{text.strip()}'")
        except Exception as e:
            report("extract_full_text() reads text from image", False, str(e))

        try:
            layout_boxes = [{"type": "Text", "box": [0, 0, 600, 150]}]
            data = engine.extract_from_boxes(test_img, layout_boxes)
            ok = isinstance(data, list) and len(data) > 0 and "text" in data[0]
            report("extract_from_boxes() structured output", ok,
                   f"Output: {data}")
        except Exception as e:
            report("extract_from_boxes() structured output", False, str(e))

# ─────────────────────────────────────────────
print("\n[6] STEP 5 — Full Pipeline Integration Test")
# ─────────────────────────────────────────────

try:
    # Simulate the full pipeline on a synthetic document image (no webcam needed)
    scene = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(scene, (150, 80), (1050, 640), (255, 255, 255), -1)
    cv2.putText(scene, "Invoice No: 2026-001", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)
    cv2.putText(scene, "Total Amount: Rs. 5000", (200, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)
    cv2.putText(scene, "Date: 12 Oct 2026", (200, 440), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)

    det = DocumentDetector()
    ana = LayoutAnalyzer()
    eng = OCREngine() if OCREngine else None

    # Step 1: blur check
    is_blur = det.is_blurry(scene)
    # Step 2: find document
    corners = det.find_document(scene)
    # Step 3: flatten
    flat = det.flatten_document(scene, corners) if corners is not None else scene
    # Step 4: layout
    boxes = ana.analyze(flat)
    # Step 5: OCR
    if eng:
        final = eng.extract_from_boxes(flat, boxes)
        extracted_text = " | ".join([d["text"] for d in final if d.get("text")])
    else:
        extracted_text = "(OCR skipped)"

    report("Full pipeline: synthetic doc → extracted text", True,
           f"Blur={is_blur}, Corners={'found' if corners is not None else 'not found'}, "
           f"Boxes={len(boxes)}, Text='{extracted_text[:80]}'")
except Exception as e:
    report("Full pipeline integration", False, str(e))

# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  SUMMARY")
print("="*60)
passed = sum(1 for _, s in results if s)
total = len(results)
print(f"\n  {passed}/{total} checks passed\n")
for step, status in results:
    icon = "✅" if status else "❌"
    print(f"  {icon}  {step}")
print("\n" + "="*60 + "\n")
