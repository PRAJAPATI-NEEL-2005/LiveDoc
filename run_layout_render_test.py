"""Run the offline pipeline on a real photo and render the 2D layout + PDF."""
import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from src.document_detector import DocumentDetector
from src.layout_analyzer import LayoutAnalyzer
from src.ocr_engine import OCREngine
from src.layout_renderer import render_page_pixmap, export_pdf

app = QApplication(sys.argv)
img_path = sys.argv[1] if len(sys.argv) > 1 else r"tested doc by user export\test 1\WhatsApp Image 2026-10-31 at 18.37.10.jpeg"
tag = sys.argv[2] if len(sys.argv) > 2 else "toc"

det, lay, ocr = DocumentDetector(), LayoutAnalyzer(), OCREngine()
frame = cv2.imread(img_path)
corners = det.find_document(frame)
flat = det.flatten_document(frame, corners) if corners is not None else frame
flat = det.preprocess_for_ocr(flat)
doc = ocr.extract_structured_document(flat, lay.analyze(flat))

page = {"elements": doc["elements"], "image_shape": doc["image_shape"], "image": flat}
os.makedirs("test_pipeline_outputs", exist_ok=True)
png = f"test_pipeline_outputs/layout_{tag}.png"
pdf = f"test_pipeline_outputs/layout_{tag}.pdf"
render_page_pixmap(page, 900).save(png)
ok = export_pdf([page], pdf, title="LiveDoc test")
print("PNG:", png)
print("PDF:", pdf, ok, os.path.getsize(pdf), "bytes")
for el in doc["elements"]:
    print(el["type"], "lines=%d" % len(el.get("lines", [])), repr((el.get("content") or "")[:60]))
