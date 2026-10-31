"""
LiveDoc Document Detector & Geometric Preprocessor
Handles:
- Real-time document edge & contour detection with dual Canny/Adaptive thresholding.
- Perspective transformation (4-point warping into flat top-down view).
- Multi-metric quality assessment: Blur/Sharpness, Glare, Lighting, Positioning guidance.
- Dynamic contrast stretching for clean OCR readability without stroke erosion.
"""

import cv2
import numpy as np


class DocumentDetector:
    def __init__(self):
        self.debug_edges = None

    def is_blurry(self, frame, threshold=22):
        """
        Check if the image is too blurry to read.
        Optimized with downscaling for sub-millisecond execution.
        """
        h, w = frame.shape[:2]
        if w > 480:
            scale = 480.0 / float(w)
            small = cv2.resize(frame, (480, int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            small = frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold

    def get_sharpness_score(self, frame):
        """Compute the numeric sharpness score of a frame."""
        h, w = frame.shape[:2]
        if w > 480:
            scale = 480.0 / float(w)
            small = cv2.resize(frame, (480, int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            small = frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def assess_frame_quality(self, frame, corners=None):
        """
        Comprehensive quality & positioning evaluation:
        - Blur / Sharpness check
        - Glare & overexposure check
        - Distance & centering assessment
        """
        orig_h, orig_w = frame.shape[:2]
        sharpness = self.get_sharpness_score(frame)

        if sharpness < 20.0:
            return {
                "status": "BLURRY",
                "message": "Hold Steady — Focus Incomplete",
                "color": (0, 140, 255),
                "score": sharpness
            }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        glare_ratio = float(np.sum(gray > 250)) / float(orig_h * orig_w)
        if glare_ratio > 0.08:
            return {
                "status": "GLARE",
                "message": "Glare Detected — Adjust Light Angle",
                "color": (0, 200, 255),
                "score": sharpness
            }

        if corners is not None:
            rect = self._order_points(corners)
            box_w = np.linalg.norm(rect[1] - rect[0])
            box_h = np.linalg.norm(rect[3] - rect[0])
            doc_area = box_w * box_h
            screen_area = orig_h * orig_w
            coverage = doc_area / screen_area

            doc_cx = float(np.mean(corners[:, 0]))
            doc_cy = float(np.mean(corners[:, 1]))
            center_x_dist = abs(doc_cx - (orig_w / 2.0)) / orig_w
            center_y_dist = abs(doc_cy - (orig_h / 2.0)) / orig_h

            if coverage < 0.18:
                return {
                    "status": "TOO_FAR",
                    "message": "Move Phone Closer to Document",
                    "color": (0, 215, 255),
                    "score": sharpness
                }
            elif center_x_dist > 0.25 or center_y_dist > 0.25:
                return {
                    "status": "OFF_CENTER",
                    "message": "Center the Document in Frame",
                    "color": (0, 215, 255),
                    "score": sharpness
                }

            return {
                "status": "LOCKED",
                "message": "Document Boundary Locked — Ready to Scan",
                "color": (0, 220, 100),
                "score": sharpness
            }

        return {
            "status": "FULL_FRAME",
            "message": "Full Document Mode — Ready",
            "color": (0, 220, 100),
            "score": sharpness
        }

    def find_document(self, frame):
        orig_h, orig_w = frame.shape[:2]
        target_w = 600
        if orig_w > target_w:
            scale = target_w / float(orig_w)
            small = cv2.resize(frame, (target_w, int(orig_h * scale)), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            small = frame

        small_h, small_w = small.shape[:2]
        small_area = small_h * small_w
        min_doc_area = small_area * 0.15
        max_doc_area = small_area * 0.98

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        v = np.median(blurred)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))
        canny = cv2.Canny(blurred, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dilated = cv2.dilate(canny, kernel, iterations=1)

        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        self.debug_edges = dilated.copy()

        for edge_img in [dilated, morph]:
            contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

            for c in contours:
                area = cv2.contourArea(c)
                if area < min_doc_area or area > max_doc_area:
                    continue

                peri = cv2.arcLength(c, True)
                for eps in [0.02, 0.03, 0.04, 0.05, 0.07]:
                    approx = cv2.approxPolyDP(c, eps * peri, True)
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        pts = (approx.reshape(4, 2) / scale).astype(int)
                        return pts

        return None

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def flatten_document(self, frame, corners):
        rect = self._order_points(corners)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
        return warped

    def preprocess_for_ocr(self, image):
        """
        Gentle contrast normalization to ensure high dynamic range without eroding thin font strokes.
        """
        if image is None or image.size == 0:
            return image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        norm = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        enhanced = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
        return enhanced
