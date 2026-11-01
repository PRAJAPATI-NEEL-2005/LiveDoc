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

    @staticmethod
    def _merge_overlapping_boxes(boxes: List[Dict[str, Any]], iou_thresh: float = 0.20, contain_thresh: float = 0.50) -> List[Dict[str, Any]]:
        """
        Merges overlapping or nested bounding boxes using Intersection over Union (IoU)
        and containment ratio to prevent duplicate table or text extractions.
        """
        if not boxes:
            return []

        merged = []
        for b in sorted(boxes, key=lambda x: (x["box"][1], x["box"][0])):
            bx1, by1, bx2, by2 = b["box"]
            b_area = max(1, (bx2 - bx1) * (by2 - by1))
            matched_idx = None

            for idx, m in enumerate(merged):
                if m.get("type") != b.get("type"):
                    continue
                mx1, my1, mx2, my2 = m["box"]
                m_area = max(1, (mx2 - mx1) * (my2 - my1))

                ix1 = max(bx1, mx1)
                iy1 = max(by1, my1)
                ix2 = min(bx2, mx2)
                iy2 = min(by2, my2)

                if ix1 < ix2 and iy1 < iy2:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    union_area = b_area + m_area - inter_area
                    iou = inter_area / float(union_area)
                    containment = inter_area / float(min(b_area, m_area))

                    if iou > iou_thresh or containment > contain_thresh:
                        matched_idx = idx
                        break

            if matched_idx is not None:
                mx1, my1, mx2, my2 = merged[matched_idx]["box"]
                merged[matched_idx]["box"] = [
                    min(bx1, mx1),
                    min(by1, my1),
                    max(bx2, mx2),
                    max(by2, my2)
                ]
                if "confidence" in merged[matched_idx] and "confidence" in b:
                    merged[matched_idx]["confidence"] = max(merged[matched_idx]["confidence"], b["confidence"])
            else:
                merged.append(b.copy())

        return merged

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

        layout_boxes = self._merge_overlapping_boxes(layout_boxes, iou_thresh=0.30, contain_thresh=0.60)
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
        # Pre-bridge faint 1-2px breaks in pencil or thin lines
        closed_thresh = cv2.morphologyEx(
            thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        )

        # Multi-scale line kernels to detect both thin/compact and wide table grids
        scale_h_fine = max(8, w // 55)
        scale_v_fine = max(8, h // 55)
        scale_h_coarse = max(14, w // 30)
        scale_v_coarse = max(14, h // 30)

        h1 = cv2.morphologyEx(closed_thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (scale_h_fine, 1)))
        v1 = cv2.morphologyEx(closed_thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale_v_fine)))
        h2 = cv2.morphologyEx(closed_thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (scale_h_coarse, 1)))
        v2 = cv2.morphologyEx(closed_thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale_v_coarse)))

        h_lines = cv2.bitwise_or(h1, h2)
        v_lines = cv2.bitwise_or(v1, v2)
        table_grid = cv2.add(h_lines, v_lines)

        # Connect subtle cell junctions and corners
        grid_connect = cv2.dilate(table_grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

        table_cnts, _ = cv2.findContours(grid_connect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_table_boxes = []
        for c in table_cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            # Table must have notable dimensions
            if bw > (w * 0.10) and bh > (h * 0.030):
                # Ignore running header rule lines at the top of the page (aspect ratio > 6.0 touching top margin)
                if by < 0.08 * h and (bw / float(max(1, bh))) > 5.5:
                    continue

                # Tight bounding box to prevent eating preceding text
                pad_x = 2
                pad_y = 2
                x1 = max(0, bx - pad_x)
                y1 = max(0, by - pad_y)
                x2 = min(w, bx + bw + pad_x)
                y2 = min(h, by + bh + pad_y)
                raw_table_boxes.append({"type": "Table", "box": [x1, y1, x2, y2]})

        # Deduplicate overlapping or nested table boxes
        table_boxes = self._merge_overlapping_boxes(raw_table_boxes, iou_thresh=0.20, contain_thresh=0.50)

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
