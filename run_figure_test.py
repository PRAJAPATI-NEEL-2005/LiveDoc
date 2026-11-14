"""Does a printed chart/diagram survive the offline pipeline -> 2D layout -> PDF?"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication
from src.layout_analyzer import LayoutAnalyzer
from src.ocr_engine import OCREngine
from src.layout_renderer import render_page_pixmap, export_pdf

app = QApplication(sys.argv)
os.makedirs("test_pipeline_outputs", exist_ok=True)

# 1. Synthetic printed page: text + bar chart (with axis labels) + caption + text
W, H = 1000, 1300
page = np.full((H, W, 3), 255, np.uint8)
F = cv2.FONT_HERSHEY_SIMPLEX


def text(s, x, y, scale=0.8, bold=1):
    cv2.putText(page, s, (x, y), F, scale, (20, 20, 20), bold, cv2.LINE_AA)


text("2.3 Quarterly Sales Analysis", 80, 110, 1.0, 2)
text("The chart below summarises the sales of three regions during the", 80, 170)
text("last four quarters. Growth was strongest in the northern region.", 80, 205)

fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=100)
x = np.arange(4)
ax.bar(x - 0.25, [30, 45, 52, 70], 0.25, label="North", color="#1f77b4")
ax.bar(x, [25, 30, 38, 41], 0.25, label="South", color="#ff7f0e")
ax.bar(x + 0.25, [20, 28, 33, 36], 0.25, label="East", color="#2ca02c")
ax.set_xticks(x)
ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
ax.set_ylabel("Sales (units)")
ax.set_title("Regional Sales")
ax.legend()
fig.tight_layout()
fig.canvas.draw()
chart = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
plt.close(fig)
ch, cw = chart.shape[:2]
x0, y0 = (W - cw) // 2, 260
page[y0:y0 + ch, x0:x0 + cw] = chart
text("Figure 2.1: Regional sales by quarter", 260, y0 + ch + 45, 0.7)
text("As the figure shows, every region grew steadily over the year, and", 80, y0 + ch + 120)
text("the northern region more than doubled its Q1 sales by Q4.", 80, y0 + ch + 155)
cv2.imwrite("test_pipeline_outputs/figure_input.png", page)

# 2. Offline pipeline
lay, ocr = LayoutAnalyzer(), OCREngine()
boxes = lay.analyze(page)
print("LAYOUT BOXES:", [(b["type"], b["box"]) for b in boxes])
doc = ocr.extract_structured_document(page, boxes)
print("\nELEMENTS:")
for el in doc["elements"]:
    print(" ", el["type"], el.get("box"), repr((el.get("content") or "")[:70]), el.get("image_path", ""))

# 3. Render
pg = {"elements": doc["elements"], "image_shape": doc["image_shape"], "image": page}
render_page_pixmap(pg, 900).save("test_pipeline_outputs/figure_layout.png")
ok = export_pdf([pg], "test_pipeline_outputs/figure_layout.pdf", title="fig test")
print("\nPDF ok:", ok)
