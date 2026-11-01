"""
LiveDoc Layout Analyzer
Decomposes document pages into semantic structural components:
- Title
- Heading / Subheading
- Paragraph
- Table
- Image / Diagram

Supports YOLOv11 deep-learning model (when models/yolo11_doc.pt is present)
and an intelligent, high-precision Computer Vision morphological layout engine as fallback.
"""

from ultralytics import YOLO
import cv2
import os
import numpy as np
from typing import List, Dict, Any


class LayoutAnalyzer:
    def __init__(self, model_path='models/yolo11_doc.pt'):
        self.use_yolo = False
        self.model_path = model_path

        if os.path.exists(model_path):
            try:
                print(f"Loading YOLO document model from {model_path}...")
                self.model = YOLO(model_path)
                self.use_yolo = True
                print("YOLO layout model loaded successfully!")
            except Exception as e:
                print(f"Failed to load YOLO model: {e}. Using CV layout analyzer.")
                self.use_yolo = False
        else:
            print(f"YOLO weights not found at {model_path}. Using high-precision CV layout analyzer.")

    def analyze(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Analyzes the flattened document to find Titles, Headings, Paragraphs, Tables, and Images.
        Returns a list of dictionaries: [{"type": "Title", "box": [x1, y1, x2, y2]}]
        """
        if self.use_yolo:
            return self._yolo_analyze(image)
        else:
            return self._cv_analyze(image)

    def _yolo_analyze(self, image: np.ndarray) -> List[Dict[str, Any]]:
        results = self.model(image)[0]
        layout_boxes = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]

            layout_boxes.append({
                "type": class_name,
                "box": [x1, y1, x2, y2],
                "confidence": float(box.conf[0])
            })

        layout_boxes.sort(key=lambda b: (b["box"][1], b["box"][0]))
        return layout_boxes

    def _cv_analyze(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        High-precision Computer Vision layout segmentation:
        1. Identifies Tables via orthogonal grid lines and cell junctions.
        2. Detects Text blocks without overlapping with tables.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Binarize with adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
        )

        # -------------------------------------------------------------
        # 1. Detect Tables (orthogonal horizontal and vertical grid lines)
        # -------------------------------------------------------------
        # Adapt kernel scales to detect both bold and thin/faint table borders
        scale_h = max(12, w // 35)
        scale_v = max(12, h // 40)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (scale_h, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale_v))

        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
        table_grid = cv2.add(h_lines, v_lines)

        # Bridge subtle 1-2px junction breaks caused by camera illumination or perspective warping
        grid_connect = cv2.dilate(table_grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

        table_cnts, _ = cv2.findContours(grid_connect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        table_boxes = []
        for c in table_cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            # Table must have notable dimensions
            if bw > (w * 0.15) and bh > (h * 0.04):
                # Tight bounding box to prevent eating preceding text
                pad_x = 4
                pad_y = 2
                x1 = max(0, bx - pad_x)
                y1 = max(0, by - pad_y)
                x2 = min(w, bx + bw + pad_x)
                y2 = min(h, by + bh + pad_y)
                table_boxes.append({"type": "Table", "box": [x1, y1, x2, y2]})

        # Sort table boxes top-to-bottom
        table_boxes.sort(key=lambda tb: tb["box"][1])

        # -------------------------------------------------------------
        # 2. Text Segmentation (Clean mask excluding tables)
        # -------------------------------------------------------------
        text_thresh = thresh.copy()
        for tb in table_boxes:
            x1, y1, x2, y2 = tb["box"]
            text_thresh[max(0, y1 - 4):min(h, y2 + 4), max(0, x1 - 4):min(w, x2 + 4)] = 0

        # Horizontal dilation to connect words in a line
        line_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, int(w * 0.03)), 2))
        lines_img = cv2.morphologyEx(text_thresh, cv2.MORPH_CLOSE, line_k)

        # Vertical dilation to group adjacent lines into paragraphs
        para_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(6, int(w * 0.015)), max(4, int(h * 0.015))))
        para_img = cv2.morphologyEx(lines_img, cv2.MORPH_CLOSE, para_k)

        cnts, _ = cv2.findContours(para_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_blocks = []

        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            if area < (w * h * 0.003):  # Ignore tiny artifacts
                continue

            # Skip page-spanning border/background contours
            if bw > (w * 0.85) and bh > (h * 0.45):
                continue

            # Classify type based on geometry and position
            if by < (h * 0.14) and bw > (w * 0.30) and bh < (h * 0.15):
                btype = "Title"
            elif bh < (h * 0.05) and bw < (w * 0.60):
                btype = "Heading"
            else:
                btype = "Paragraph"

            pad_x = 4
            pad_y = 2
            px1 = max(0, bx - pad_x)
            py1 = max(0, by - pad_y)
            px2 = min(w, bx + bw + pad_x)
            py2 = min(h, by + bh + pad_y)

            text_blocks.append({
                "type": btype,
                "box": [px1, py1, px2, py2]
            })

        all_blocks = table_boxes + text_blocks
        if not all_blocks:
            return [{"type": "Paragraph", "box": [0, 0, w, h]}]

        all_blocks.sort(key=lambda b: (b["box"][1], b["box"][0]))
        return all_blocks
