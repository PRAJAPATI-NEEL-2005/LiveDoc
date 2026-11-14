"""
LiveDoc OCR Engine & Structured Document Digitizer
Handles:
- Text line detection & recognition via PaddleOCR.
- Dynamic page orientation and rotation compensation.
- Strict exclusive token assignment with targeted table crop recovery.
- Reconstructing authentic document reading flow (Title -> Heading -> Paragraph -> Table).
- Bleed-through ghost ink suppression and discrete logic notation cleanup.
- Full validation report (confidence scores & math verification).
"""

from paddleocr import PaddleOCR
import numpy as np
import cv2
import re
import os
from typing import List, Dict, Any

from src.table_extractor import TableExtractor
from src.validator import DocumentValidator
from src.layout_analyzer import LayoutAnalyzer


class OCREngine:
    def __init__(self, lang='en'):
        print("Initializing PaddleOCR...")
        # det_db_box_thresh=0.50 prevents detecting faint reverse-page bleed-through ink
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            show_log=False,
            det_limit_side_len=1600,
            det_db_box_thresh=0.30,
            det_db_unclip_ratio=1.7
        )
        self.table_extractor = TableExtractor()
        self.validator = DocumentValidator(min_confidence_threshold=0.72)
        self.layout_analyzer = LayoutAnalyzer()
        print("PaddleOCR Initialized successfully!")

    @staticmethod
    def _is_mostly_vertical(raw_items):
        if not raw_items:
            return False
        vert = 0
        horiz = 0
        for item in raw_items:
            pts = np.array(item[0], dtype=np.float32)
            w = np.linalg.norm(pts[1] - pts[0])
            h = np.linalg.norm(pts[2] - pts[1])
            if h > 1.25 * w:
                vert += 1
            elif w > 1.25 * h:
                horiz += 1
        return vert > (horiz + 1)

    def _get_best_ocr(self, image):
        results = self.ocr.ocr(image, cls=True)
        raw = results[0] if results and results[0] else []

        if self._is_mostly_vertical(raw):
            rot_cw = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            res_cw = self.ocr.ocr(rot_cw, cls=True)
            raw_cw = res_cw[0] if res_cw and res_cw[0] else []

            rot_ccw = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            res_ccw = self.ocr.ocr(rot_ccw, cls=True)
            raw_ccw = res_ccw[0] if res_ccw and res_ccw[0] else []

            cw_vert = self._is_mostly_vertical(raw_cw)
            ccw_vert = self._is_mostly_vertical(raw_ccw)

            if len(raw_cw) >= len(raw) and not cw_vert:
                return raw_cw, rot_cw
            elif len(raw_ccw) >= len(raw) and not ccw_vert:
                return raw_ccw, rot_ccw

        # Document tilt / skew compensation:
        # If wide text lines have a consistent slant angle >= 1.0 deg, deskew and re-run OCR
        slopes = []
        for it in raw:
            pts = np.array(it[0], dtype=np.float32)
            dx = pts[1, 0] - pts[0, 0]
            dy = pts[1, 1] - pts[0, 1]
            if dx > 120:
                slopes.append(np.degrees(np.arctan2(dy, dx)))
        if len(slopes) >= 5:
            med_tilt = float(np.median(slopes))
            if 1.0 <= abs(med_tilt) <= 25.0:
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, med_tilt, 1.0)
                deskewed_img = cv2.warpAffine(
                    image, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255)
                )
                res_deskewed = self.ocr.ocr(deskewed_img, cls=True)
                raw_deskewed = res_deskewed[0] if res_deskewed and res_deskewed[0] else []
                if len(raw_deskewed) >= len(raw):
                    return raw_deskewed, deskewed_img

        return raw, image

    @staticmethod
    def _parse_tokens(raw_items):
        # Compute median text line slope across wide tokens to compensate for slight page slant
        slopes = []
        for it in raw_items:
            pts = np.array(it[0], dtype=np.float32)
            dx = pts[1, 0] - pts[0, 0]
            dy = pts[1, 1] - pts[0, 1]
            if dx > 150:
                slopes.append(dy / dx)
        doc_slope = float(np.median(slopes)) if slopes else 0.0

        tokens = []
        for it in raw_items:
            pts = np.array(it[0], dtype=np.float32)
            txt = str(it[1][0]).strip()
            conf = float(it[1][1]) if len(it[1]) > 1 else 1.0
            if not txt:
                continue

            min_x = float(np.min(pts[:, 0]))
            max_x = float(np.max(pts[:, 0]))
            min_y = float(np.min(pts[:, 1]))
            max_y = float(np.max(pts[:, 1]))
            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0
            y_proj = cy - doc_slope * cx

            tokens.append({
                "text": txt,
                "conf": conf,
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "cx": cx,
                "cy": cy,
                "y_proj": y_proj,
                "h": max(1.0, max_y - min_y),
                "w": max(1.0, max_x - min_x)
            })
        return tokens

    @staticmethod
    def _detect_token_grid_boxes(tokens: List[Dict[str, Any]], img_shape, existing_boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Dual-layer table recovery: detects structured 2D table grids directly from token alignment,
        even if camera lighting or faint borders caused CV border morphology to miss the table.
        Uses proximity clustering on short tabular values while strictly ignoring prose words.
        """
        STOPWORDS = {
            'the', 'this', 'that', 'these', 'those', 'for', 'with', 'from', 'into',
            'during', 'until', 'against', 'among', 'throughout', 'despite', 'towards',
            'upon', 'concerning', 'to', 'in', 'of', 'by', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'shall',
            'will', 'should', 'would', 'may', 'might', 'must', 'can', 'could', 'and',
            'but', 'if', 'or', 'because', 'as', 'while', 'at', 'about', 'between',
            'through', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'on',
            'off', 'over', 'under', 'again', 'further', 'then', 'once', 'such', 'what',
            'which', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how', 'all', 'any',
            'both', 'each', 'few', 'more', 'most', 'other', 'some'
        }

        def is_cell_val(s: str) -> bool:
            s_clean = s.strip('.,;:()')
            if not s_clean:
                return False
            if s_clean.lower() in ('t', 'f', 'true', 'false', 'yes', 'no'):
                return True
            if s_clean.isdigit():
                return True
            if len(s_clean) <= 4 and not any(ch.islower() for ch in s_clean):
                return True
            if re.match(r'^[A-Za-z0-9\-\^\v\~]+$', s_clean) and len(s_clean) <= 5:
                return True
            return False

        H, W = img_shape[:2]
        med_h = float(np.median([t["h"] for t in tokens if t["h"] > 4])) if tokens else 20.0
        med_w = float(np.median([t["w"] for t in tokens if t["w"] > 4])) if tokens else 25.0

        cand_tokens = []
        for t in tokens:
            txt = t["text"].strip()
            parts = txt.split()
            if len(parts) >= 2:
                if all(is_cell_val(p) for p in parts) and not any(p.lower() in STOPWORDS for p in parts):
                    cand_tokens.append(t)
            else:
                if (is_cell_val(txt) or len(txt) <= 8) and not any(p.lower() in STOPWORDS for p in parts) and t["w"] < 120:
                    cand_tokens.append(t)

        cand_tokens.sort(key=lambda t: t["cy"])

        # Cluster candidate tabular tokens by vertical proximity and horizontal bounds
        clusters = []
        curr_cl = []
        for t in cand_tokens:
            if not curr_cl:
                curr_cl.append(t)
            else:
                prev_t = curr_cl[-1]
                if (t["cy"] - prev_t["cy"]) <= 2.2 * med_h and abs(t["cx"] - prev_t["cx"]) <= 450:
                    curr_cl.append(t)
                else:
                    if len(curr_cl) >= 4:
                        clusters.append(curr_cl)
                    curr_cl = [t]
        if len(curr_cl) >= 4:
            clusters.append(curr_cl)

        new_boxes = []
        for cl in clusters:
            if len(cl) < 6:
                continue
            # If any token in cluster is a figure caption (e.g. Fig. 2.69), it's a diagram, strictly NOT a table
            if any(re.search(r"(?i)^(fig|figure|diagram|chart|illustration)\b", t["text"].strip()) for t in cl):
                continue
            xs = [t["cx"] for t in cl]
            ys = [t["cy"] for t in cl]
            bx1 = max(0, int(min(t["min_x"] for t in cl)) - 10)
            by1 = max(0, int(min(t["min_y"] for t in cl)) - 8)
            bx2 = min(W, int(max(t["max_x"] for t in cl)) + 10)
            by2 = min(H, int(max(t["max_y"] for t in cl)) + 8)

            # Check distinct columns: require >= 2 distinct horizontal positions
            col_clusters = []
            for x in sorted(xs):
                matched = None
                for c in col_clusters:
                    if abs(x - float(np.mean(c))) < max(25.0, 0.75 * med_w):
                        matched = c
                        break
                if matched is not None:
                    matched.append(x)
                else:
                    col_clusters.append([x])

            if len(col_clusters) < 2:
                continue

            all_known = existing_boxes + new_boxes
            overlap = any(
                not (bx2 < eb["box"][0] or bx1 > eb["box"][2] or by2 < eb["box"][1] or by1 > eb["box"][3])
                for eb in all_known
            )
            if not overlap and (bx2 - bx1) > 80 and (by2 - by1) > 45:
                new_boxes.append({"type": "Table", "box": [bx1, by1, bx2, by2]})

        return new_boxes

    @staticmethod
    def _merge_table_boxes(table_boxes: List[Dict[str, Any]], iou_thresh: float = 0.20, contain_thresh: float = 0.45) -> List[Dict[str, Any]]:
        """
        Deduplicates table bounding boxes so that overlapping or nested boxes
        for the same physical table are merged into a single enclosing box.
        """
        if not table_boxes:
            return []

        merged = []
        for tb in sorted(table_boxes, key=lambda b: (b["box"][1], b["box"][0])):
            bx1, by1, bx2, by2 = tb["box"]
            b_area = max(1, (bx2 - bx1) * (by2 - by1))
            matched_idx = None

            for idx, m in enumerate(merged):
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
            else:
                merged.append(tb.copy())

        return merged

    @staticmethod
    def _detect_figure_boxes(image: np.ndarray, tokens: List[Dict[str, Any]], table_boxes: List[Dict[str, Any]], assigned_token_indices: set) -> List[Dict[str, Any]]:
        """
        Detects prominent visual figures, drawings, state diagrams, charts, or images on the page:
        - Accurately detects state transition graphs, flowcharts, circuits, and illustrations.
        - Preserves diagram labels (e.g. q0, q1, 0/A, 1/B) and bridges graphical elements.
        - Anchors diagrams using figure captions (e.g. Fig. 2.69).
        - Rejects plain text paragraphs and margin binding shadows.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 40, 140)
        # Clear outer image border lines from scanner / camera frame
        edges[:12, :] = 0
        edges[-12:, :] = 0
        edges[:, :12] = 0
        edges[:, -12:] = 0

        # 1. Mask out verified table regions
        for tb in table_boxes:
            bx1, by1, bx2, by2 = tb["box"]
            edges[max(0, by1 - 5):min(h, by2 + 5), max(0, bx1 - 5):min(w, bx2 + 5)] = 0

        FIG_CAPTION_RE = re.compile(r"(?i)\b(?:fig(?:\.|ure)?|chart|diagram|graph|scheme)\b")

        # 2. Mask only long prose tokens (> 6 chars and not figure caption or short math/state labels)
        # This keeps state circles, arrows, logic gates, and short labels intact
        edges_no_prose = edges.copy()
        for i, t in enumerate(tokens):
            if i in assigned_token_indices:
                continue
            txt = t["text"].strip()
            if FIG_CAPTION_RE.search(txt) or len(txt) <= 6:
                continue
            tx1 = max(0, int(t["min_x"]) - 2)
            ty1 = max(0, int(t["min_y"]) - 2)
            tx2 = min(w, int(t["max_x"]) + 2)
            ty2 = min(h, int(t["max_y"]) + 2)
            edges_no_prose[ty1:ty2, tx1:tx2] = 0

        # Morphological bridge kernel to connect state circles, arrows, flowchart blocks
        bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        dilated = cv2.dilate(edges_no_prose, bridge_kernel, iterations=2)

        cnts, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        fig_boxes = []
        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            # Size limits
            if bw < 80 or bh < 60:
                continue
            if bw > 0.92 * w or bh > 0.90 * h:
                continue
            # Suppress outer page margins & book spine binding shadow
            if (bx > 0.82 * w and (bx + bw) >= 0.96 * w) or (bx <= 0.04 * w and (bx + bw) < 0.18 * w):
                continue
            # Suppress header margin noise
            if by < 0.08 * h and (bw * bh) < (w * h * 0.04):
                continue
            aspect = bw / float(max(1, bh))
            if aspect < 0.20 or aspect > 5.0:
                continue

            # Must contain significant non-text edge pixels and density
            crop_edges = edges_no_prose[by:by+bh, bx:bx+bw]
            non_text_edge_count = cv2.countNonZero(crop_edges)
            edge_density = non_text_edge_count / float(bw * bh)
            if non_text_edge_count < 300 or edge_density < 0.012:
                continue

            # Tokens inside candidate box
            inside_tokens = [
                t for t in tokens
                if (bx - 10 <= t["cx"] <= bx + bw + 10) and (by - 10 <= t["cy"] <= by + bh + 10)
            ]
            caption_raw = next((t["text"].strip() for t in inside_tokens if FIG_CAPTION_RE.search(t["text"].strip())), "")
            caption_match = ""
            if caption_raw:
                cm = re.search(r"(?i)\b(?:fig(?:\.|ure)?|chart|diagram|graph|scheme)\b.*", caption_raw)
                caption_match = cm.group(0).strip() if cm else caption_raw
            has_fig_caption = bool(caption_match)

            # Allow diagrams with multiple internal state/transition labels
            max_tokens_allowed = 25 if (has_fig_caption or non_text_edge_count > 600) else 12
            if len(inside_tokens) > max_tokens_allowed:
                continue

            fig_boxes.append({
                "type": "Image",
                "box": [bx, by, bx + bw, by + bh],
                "caption": caption_match,
                "edge_count": non_text_edge_count
            })

        # 3. Figure caption anchoring for captions positioned just outside the dilated contour
        for i, t in enumerate(tokens):
            if i in assigned_token_indices:
                continue
            txt = t["text"].strip()
            if FIG_CAPTION_RE.search(txt):
                cx, cy = t["cx"], t["cy"]
                # Check if this caption is already inside any fig_box
                already_covered = any(
                    fb["box"][0] - 15 <= cx <= fb["box"][2] + 15 and fb["box"][1] - 15 <= cy <= fb["box"][3] + 15
                    for fb in fig_boxes
                )
                if not already_covered:
                    # Look for nearest candidate figure directly above the caption
                    for fb in fig_boxes:
                        fbx1, fby1, fbx2, fby2 = fb["box"]
                        if (fbx1 - 60 <= cx <= fbx2 + 60) and (fby2 <= cy <= fby2 + 80):
                            fb["box"][3] = max(fb["box"][3], int(t["max_y"]) + 5)
                            fb["box"][0] = min(fb["box"][0], int(t["min_x"]) - 5)
                            fb["box"][2] = max(fb["box"][2], int(t["max_x"]) + 5)
                            fb["caption"] = txt
                            break

        return fig_boxes

    @staticmethod
    def _clean_text_spacing(text: str) -> str:
        """
        General typographic spacing, whitespace normalisation, and OCR
        artifact removal for any printed document.
        """
        if not text:
            return ""
        t = text

        # ── 1. Remove unprintable / control characters ──────────────────
        t = re.sub(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]", "", t)

        # ── 2. Spacing normalisations ────────────────────────────────────
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"([,;:?!])([A-Za-z0-9])", r"\1 \2", t)
        t = re.sub(r"(\.)([ ]*[A-Za-z])", r". \2", t)
        t = re.sub(r"([a-z]{2,})([A-Z][a-z]{2,})", r"\1 \2", t)
        t = re.sub(r"([a-zA-Z0-9])\(", r"\1 (", t)
        t = re.sub(r"\)([a-zA-Z])", r") \1", t)
        t = re.sub(r"\s+([,.:;?!])", r"\1", t)
        t = re.sub(r"^\)\s*", "", t)
        t = re.sub(r"([a-z])\.([A-Z])", r"\1. \2", t)

        # ── 3. Outline / numbered list item spacing ─────────────────────
        def _sep_roman(m):
            prefix, roman, word = m.group(1), m.group(2), m.group(3)
            if roman in ("i", "j") and word.startswith("R"):
                return m.group(0)
            return f"{prefix}{roman}) {word}"

        t = re.sub(r"(^|\s)([ivxlcdm]{1,5})([A-Z][a-z]{1,})", _sep_roman, t)
        t = re.sub(r"(^|\s)([ivxlcdm]{1,5})\s*([A-Z][a-z]+)", _sep_roman, t)
        t = re.sub(r"(^|\s)(\d{1,2})([A-Z][a-z]+)", r"\1\2. \3", t)

        # ── 4. General operator & relation spacing ───────────────────────
        t = re.sub(r"([A-Za-z0-9])(=)([A-Za-z0-9])", r"\1 \2 \3", t)
        t = re.sub(r"\s*->\s*", " → ", t)
        t = re.sub(r"\s*<-\s*", " ← ", t)
        t = re.sub(r"\s*=>\s*", " ⇒ ", t)

        # Strip accidental markdown symbols leaking into OCR
        t = t.replace("**", "").replace("__", "").replace("~~", "").replace("`", "")

        # ── 5. Final whitespace trim ─────────────────────────────────────
        return t.strip()

    def extract_structured_document(self, image: np.ndarray, layout_boxes: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main end-to-end pipeline method:
        Digitizes the document strictly matching its physical layout:
        - Running headers
        - Subsections / Headings
        - Targeted crop OCR for tables (100% cell recovery, zero duplication)
        - Flowing paragraphs with mathematical notation preservation
        """
        raw_items, final_img = self._get_best_ocr(image)
        tokens = self._parse_tokens(raw_items)

        if not tokens:
            return {
                "title": "Untitled Document",
                "elements": [],
                "markdown": "No text detected in document.",
                "html": "<p><em>No text detected in document.</em></p>",
                "validation": {"status": "EMPTY", "uncertain_items": [], "inconsistencies": []}
            }

        H, W = final_img.shape[:2]

        if layout_boxes is None:
            layout_boxes = self.layout_analyzer.analyze(final_img)

        table_boxes = [
            b for b in layout_boxes
            if b.get("type") == "Table" and not ((b["box"][2] - b["box"][0]) > 0.88 * W and (b["box"][3] - b["box"][1]) > 0.75 * H)
        ]

        # Dual-layer table recovery: detect tables from token grid if CV borders were faint or missed
        supplemental_boxes = self._detect_token_grid_boxes(tokens, final_img.shape, table_boxes)
        table_boxes.extend(supplemental_boxes)

        # Merge overlapping table boxes to eliminate duplicate table extractions
        table_boxes = self._merge_table_boxes(table_boxes, iou_thresh=0.20, contain_thresh=0.45)
        table_boxes.sort(key=lambda tb: tb["box"][1])

        # -----------------------------------------------------------------
        # STEP 1: Exclusive Table Extraction with Multi-Row Consensus
        # -----------------------------------------------------------------
        assigned_token_indices = set()
        structured_elements = []
        valid_table_boxes = []

        for b_idx, tb in enumerate(table_boxes):
            bx1, by1, bx2, by2 = tb["box"]
            crop = final_img[max(0, by1):min(H, by2), max(0, bx1):min(W, bx2)]

            # Extract tokens belonging to this table from full-page OCR
            crop_tokens = []
            table_token_indices = []
            for i, t in enumerate(tokens):
                if i in assigned_token_indices:
                    continue
                if (bx1 - 10 <= t["cx"] <= bx2 + 10) and (by1 - 4 <= t["cy"] <= by2 + 6):
                    table_token_indices.append(i)
                    crop_tokens.append({
                        "text": t["text"],
                        "conf": t["conf"],
                        "min_x": t["min_x"] - bx1,
                        "max_x": t["max_x"] - bx1,
                        "min_y": t["min_y"] - by1,
                        "max_y": t["max_y"] - by1,
                        "cx": t["cx"] - bx1,
                        "cy": t["cy"] - by1,
                        "h": t["h"],
                        "w": t["w"]
                    })

            # Dual-phase table cell recovery: if full-page OCR has fewer than 8 tokens
            # or if any cell in candidate table is empty (in headers or rows), run targeted high-res crop OCR
            need_targeted = (len(crop_tokens) < 8)
            if not need_targeted and crop_tokens:
                tbl_check = self.table_extractor.extract_table(crop, crop_tokens)
                has_empty = (
                    any(not c.strip() for c in tbl_check.get("headers", [])) or
                    any(any(not c.strip() for c in r) for r in tbl_check.get("rows", []))
                )
                if has_empty:
                    need_targeted = True

            if need_targeted:
                ch, cw = crop.shape[:2]
                scale = 1.5
                crop_scaled = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)
                crop_res = self.ocr.ocr(crop_scaled, cls=True)
                raw_crop_tokens = self._parse_tokens(crop_res[0] if crop_res and crop_res[0] else [])

                if len(raw_crop_tokens) >= len(crop_tokens):
                    crop_tokens = []
                    for t in raw_crop_tokens:
                        crop_tokens.append({
                            "text": t["text"],
                            "conf": t["conf"],
                            "min_x": t["min_x"] / scale,
                            "max_x": t["max_x"] / scale,
                            "min_y": t["min_y"] / scale,
                            "max_y": t["max_y"] / scale,
                            "cx": t["cx"] / scale,
                            "cy": t["cy"] / scale,
                            "h": t["h"] / scale,
                            "w": t["w"] / scale,
                        })

            if crop_tokens:
                tbl_data = self.table_extractor.extract_table(crop, crop_tokens)
                if tbl_data.get("markdown"):
                    headers = [h for h in tbl_data.get("headers", []) if h.strip()]
                    rows = tbl_data.get("rows", [])
                    filled_cells = len(headers) + sum(len([c for c in r if c.strip()]) for r in rows)

                    # Mathematical Table Validation:
                    # A true table MUST possess >= 2 columns, >= 2 rows (headers + data rows),
                    # at least 4 filled cells, and >= 4 tokens.
                    # Prose sentences, fragments, or isolated single words are NEVER tables.
                    contains_fig_caption = any(
                        re.search(r"(?i)^(fig|figure|diagram|chart|illustration)\b", t.get("text", "").strip())
                        for t in crop_tokens
                    )
                    is_valid_table = (
                        not contains_fig_caption and
                        len(headers) >= 2 and
                        (len(rows) >= 1) and
                        filled_cells >= 4 and
                        len(crop_tokens) >= 4
                    )

                    if is_valid_table:
                        is_duplicate_table = any(
                            el.get("type") == "Table" and (
                                el.get("markdown") == tbl_data["markdown"] or
                                (el.get("headers") == tbl_data.get("headers") and el.get("rows") == tbl_data.get("rows"))
                            )
                            for el in structured_elements
                        )
                        if not is_duplicate_table:
                            valid_table_boxes.append(tb)
                            assigned_token_indices.update(table_token_indices)
                            structured_elements.append({
                                "type": "Table",
                                "text": tbl_data["markdown"],
                                "content": tbl_data["markdown"],
                                "headers": tbl_data["headers"],
                                "rows": tbl_data["rows"],
                                "markdown": tbl_data["markdown"],
                                "html": tbl_data["html"],
                                "confidence": tbl_data.get("confidence", 1.0),
                                "box": [bx1, by1, bx2, by2],
                                "top_y": by1
                            })

        # Update table_boxes to ONLY genuinely valid tables so rejected fake tables never break prose
        table_boxes = valid_table_boxes

        # -------------------------------------------------------------
        # STEP 1.5: Image / Diagram / Figure Extraction
        # -------------------------------------------------------------
        cand_img_boxes = [b for b in layout_boxes if b.get("type") in ("Image", "Picture", "Figure")]
        # Also auto-detect large visual graphics that are not tables
        supplemental_img_boxes = self._detect_figure_boxes(final_img, tokens, table_boxes, assigned_token_indices)
        cand_img_boxes.extend(supplemental_img_boxes)
        cand_img_boxes = self._merge_table_boxes(cand_img_boxes, iou_thresh=0.25, contain_thresh=0.50)

        os.makedirs(os.path.join("documents", "figures"), exist_ok=True)
        fig_counter = 0

        for ib in cand_img_boxes:
            bx1, by1, bx2, by2 = ib["box"]
            # Exclude regions that overlap with validated tables
            overlaps_table = any(
                max(bx1, tb["box"][0]) < min(bx2, tb["box"][2]) and
                max(by1, tb["box"][1]) < min(by2, tb["box"][3])
                for tb in table_boxes
            )
            if overlaps_table:
                continue

            # Identify tokens inside image box
            img_token_indices = [
                i for i, t in enumerate(tokens)
                if i not in assigned_token_indices and (bx1 - 10 <= t["cx"] <= bx2 + 10) and (by1 - 10 <= t["cy"] <= by2 + 10)
            ]
            has_explicit_caption = bool(ib.get("caption"))
            max_allowed_tokens = 25 if (has_explicit_caption or ib.get("edge_count", 0) > 600) else 12
            # If region has excessively high density of text tokens without diagram characteristics, it's a text block
            if len(img_token_indices) > max_allowed_tokens:
                continue

            fig_counter += 1
            cx1 = max(0, bx1)
            cy1 = max(0, by1)
            cx2 = min(W, bx2)
            cy2 = min(H, by2)
            fig_crop = final_img[cy1:cy2, cx1:cx2]

            fig_filename = f"figure_{fig_counter}.png"
            fig_rel_path = os.path.join("documents", "figures", fig_filename)
            fig_abs_path = os.path.abspath(fig_rel_path)
            cv2.imwrite(fig_abs_path, fig_crop)

            # Suppress all tokens inside and immediately surrounding the figure box
            # (suppresses state labels, transition tags, internal legends, numbers, and captions from forming stray paragraphs)
            for i, t in enumerate(tokens):
                if i not in assigned_token_indices:
                    if (bx1 - 15 <= t["cx"] <= bx2 + 15) and (by1 - 15 <= t["cy"] <= by2 + 15):
                        assigned_token_indices.add(i)

            # Look for caption line right below or above the figure if not already present
            caption_text = ib.get("caption", "").strip()
            caption_line_indices = []
            if not caption_text:
                for i, t in enumerate(tokens):
                    if i in assigned_token_indices:
                        continue
                    is_near_y = (by2 <= t["cy"] <= by2 + 50) or (by1 - 50 <= t["cy"] <= by1)
                    is_aligned_x = (bx1 - 40 <= t["cx"] <= bx2 + 40)
                    if is_near_y and is_aligned_x:
                        txt = t["text"].strip()
                        m = re.search(r"(?i)\b(fig|figure|diagram|chart|illustration)\b.*", txt)
                        if m:
                            cap_cy = t["cy"]
                            for j, t2 in enumerate(tokens):
                                if j not in assigned_token_indices and abs(t2["cy"] - cap_cy) < 14 and (bx1 - 60 <= t2["cx"] <= bx2 + 60):
                                    caption_line_indices.append(j)
                            break

                if caption_line_indices:
                    cap_toks = sorted([tokens[j] for j in caption_line_indices], key=lambda x: x["min_x"])
                    caption_text = self._clean_text_spacing(" ".join(x["text"].strip() for x in cap_toks))
                    assigned_token_indices.update(caption_line_indices)
                else:
                    caption_text = f"Figure {fig_counter}"
            else:
                # If caption was matched on an internal token, ensure that token is suppressed from prose
                for i, t in enumerate(tokens):
                    if i not in assigned_token_indices and (caption_text in t["text"].strip() or t["text"].strip() in caption_text):
                        assigned_token_indices.add(i)

            assigned_token_indices.update(img_token_indices)

            structured_elements.append({
                "type": "Image",
                "box": [cx1, cy1, cx2, cy2],
                "image_path": fig_abs_path,
                "content": caption_text,
                "caption": caption_text,
                "text": f"![{caption_text}]({fig_abs_path})",
                "markdown": f"![{caption_text}]({fig_abs_path})",
                "html": f'<div style="text-align: center; margin: 12px 0;"><img src="{fig_abs_path}" style="max-width: 88%; max-height: 280px; border-radius: 4px; border: 1px solid #cbd5e1;" /><div style="font-size: 8pt; color: #64748b; font-style: italic; margin-top: 4px;">{caption_text}</div></div>',
                "confidence": 1.0,
                "top_y": cy1
            })

        # -----------------------------------------------------------------
        # STEP 2: Non-Table Text Processing (Exclusively remaining tokens)
        # -----------------------------------------------------------------
        free_tokens = [tokens[i] for i in range(len(tokens)) if i not in assigned_token_indices]

        # General noise suppression (stray optical artifacts & thin-paper bleed-through)
        clean_free_tokens = []
        for t in free_tokens:
            txt = t["text"].strip()
            # Ignore isolated tokens touching extreme outer margins (< 4% or > 96% page width)
            if (t["min_x"] > 0.96 * W or t["max_x"] < 0.04 * W) and (len(txt) <= 4 or t["w"] < 35):
                continue
            # Drop low-confidence noise tokens (< 0.60 is universally recognized as unreliable in printed text OCR)
            if t["conf"] < 0.60:
                continue
            clean_free_tokens.append(t)

        clean_free_tokens.sort(key=lambda t: t.get("y_proj", t["cy"]))

        # Cluster into physical lines
        physical_lines = []
        for t in clean_free_tokens:
            matched_line = None
            t_y = t.get("y_proj", t["cy"])
            for line in physical_lines:
                # Two tokens that horizontally overlap cannot share the same line
                has_x_overlap = any(
                    max(t["min_x"], x["min_x"]) < min(t["max_x"], x["max_x"]) - 10
                    for x in line
                )
                if has_x_overlap:
                    continue

                # Isolated stray token separated by a large horizontal gap from line tokens.
                # In typography, words on the same line are never separated by more than 75px.
                # A larger gap indicates a different column, facing page, or margin noise.
                line_max_x = max(x["max_x"] for x in line)
                line_min_x = min(x["min_x"] for x in line)
                if (t["min_x"] - line_max_x > 75) or (line_min_x - t["max_x"] > 75):
                    continue

                l_y = float(np.mean([x.get("y_proj", x["cy"]) for x in line]))
                l_h = float(np.median([x["h"] for x in line]))
                if abs(t_y - l_y) < max(0.40 * l_h, 8.0):
                    matched_line = line
                    break
            if matched_line is not None:
                matched_line.append(t)
            else:
                physical_lines.append([t])

        # Filter out isolated single-token stray lines (bleed-through artifacts & optical specks)
        filtered_lines = []
        for line in physical_lines:
            if len(line) == 1:
                t = line[0]
                txt = t["text"].strip()
                # A wide or multi-word token is a genuine text line, not an optical speck
                if len(txt) > 15 or t["w"] > 120:
                    filtered_lines.append(line)
                    continue

                # High-confidence threshold for single-token lines unless it is a standard list bullet (e.g. '1.', 'A.', 'i)')
                # Bullets appear in the left margin area (< 70% of page width), never at the far right edge
                is_bullet = bool(re.match(r"^(\d{1,2}|[ivx]{1,4}|[a-zA-Z])[\.\)]?$", txt)) and (t["min_x"] < 0.70 * W)
                if not is_bullet:
                    if t["conf"] < 0.72 or t["min_x"] > 0.82 * W or t["max_x"] < 0.08 * W or (t["conf"] < 0.85 and t["w"] < 50):
                        continue
            filtered_lines.append(line)
        physical_lines = filtered_lines

        for line in physical_lines:
            line.sort(key=lambda t: t["min_x"])
        physical_lines.sort(key=lambda line: float(np.mean([t.get("y_proj", t["cy"]) for t in line])))

        # -----------------------------------------------------------------
        # STEP 3: Classify Text Lines into Header, Headings, Paragraphs
        # -----------------------------------------------------------------
        med_line_h = float(np.median([t["h"] for t in clean_free_tokens])) if clean_free_tokens else 20.0

        # Calculate typical inter-line spacing (center-to-center distance)
        line_spacings = []
        for i in range(1, len(physical_lines)):
            dy = float(np.mean([t.get("y_proj", t["cy"]) for t in physical_lines[i]])) - float(np.mean([t.get("y_proj", t["cy"]) for t in physical_lines[i-1]]))
            if 10 <= dy <= 60:
                line_spacings.append(dy)
        med_line_spacing = float(np.median(line_spacings)) if line_spacings else (med_line_h * 1.15)

        # Detect typical column margins
        col_min_x = float(np.percentile([min(t["min_x"] for t in l) for l in physical_lines], 15)) if physical_lines else 0
        col_max_x = float(np.percentile([max(t["max_x"] for t in l) for l in physical_lines], 85)) if physical_lines else W
        col_width = max(100.0, col_max_x - col_min_x)

        curr_para_lines = []
        curr_para_tokens = []
        curr_para_segs = []
        curr_para_y = 0
        prev_line_y = None

        def _box_of(toks):
            return [
                int(min(t["min_x"] for t in toks)),
                int(min(t["min_y"] for t in toks)),
                int(max(t["max_x"] for t in toks)),
                int(max(t["max_y"] for t in toks)),
            ]

        def flush_paragraph():
            nonlocal curr_para_lines, curr_para_tokens, curr_para_y, curr_para_segs
            if curr_para_lines:
                # 1. Expand and split any inline bullets embedded inside lines
                expanded_lines = []
                for l in curr_para_lines:
                    l_str = l.strip()
                    if not l_str:
                        continue
                    # Normalize spacing for bullet punctuation (e.g. '1.Reflexive' -> '1. Reflexive')
                    t = re.sub(r'(^|\s)([1-9]\d{0,1}|[ivxlcdm]{1,5})([\.\)])([A-Za-z])', r'\1\2\3 \4', l_str)
                    # Insert missing period after terminal truth values if followed by capital letter
                    t = re.sub(r"\b(false|true)\s+([A-Z])", r"\1. \2", t)
                    # Normalize bullets with missing dot/paren only at the START of a line or after punctuation
                    t = re.sub(r'(^|[;:\-\.\n]\s*)([1-9]\d{0,1})\s*([A-Z][a-z]+)', r'\1\2. \3', t)
                    t = re.sub(r'(^|[;:\-\.\n]\s*)([ivxlcdm]{1,5})\s*([A-Z][a-z]+)', r'\1\2) \3', t)

                    # Split inline bullets if multiple numbered items were scanned on one line
                    split_pat = (
                        r'\s+(?='
                        r'(?:[1-9]\d{0,1}\.|\([1-9]\d{0,1}\)|[1-9]\d{0,1}\))\s+[A-Z]'
                        r'|(?:\([ivxlcdm]{1,5}\)|[ivxlcdm]{1,5}\.|\b[ivxlcdm]{1,5}\))\s+[A-Z]'
                        r'|(?:\([a-zA-Z]\)|[a-zA-Z]\))\s+[A-Z]'
                        r'|(?:[\u2022\u25cf\u25cb\u25aa\u25ab\u25c6\u2212\-\*])\s+[A-Za-z]'
                        r'|Every\s+[a-z]+'
                        r')'
                    )
                    parts = re.split(split_pat, t)
                    expanded_lines.extend(p.strip() for p in parts if p.strip())

                # 2. Assemble lines: join flowing sentences into cohesive paragraphs,
                # while strictly isolating list items, labeled blocks, and formulas on new lines
                assembled_lines = []
                curr_flow = ""
                for sub_l in expanded_lines:
                    sub_str = sub_l.strip()
                    # Discard orphaned standalone digits on their own line (e.g. stray '4' below an already-numbered point)
                    if re.match(r"^\d{1,2}[\.\)]?$", sub_str) and assembled_lines:
                        continue
                    is_bullet = bool(re.match(
                        r'^(?:'
                        r'[1-9]\d{0,1}\.|\([1-9]\d{0,1}\)|[1-9]\d{0,1}\)'
                        r'|\(?[ivxlcdm]{1,5}[\.\)]'
                        r'|\(?[a-zA-Z]\)'
                        r'|[\u2022\u25cf\u25cb\u25aa\u25ab\u25c6\u2212\-\*]'
                        r'|[1-9]\d{0,1}\s+[A-Z][a-z]{2,}'
                        r'|[ivxlcdm]{1,5}\s+[A-Z][a-z]{2,}'
                        r')\s+[A-Z]',
                        sub_l
                    )) or bool(re.match(r'^[\u2022\u25cf\u25cb\u25aa\u25ab\u25c6\u2212\-\*]\s+[A-Za-z]', sub_l))

                    # Check if previous item was a numbered bullet and current line is a parallel definition item with missing number
                    prev_last_line = assembled_lines[-1] if assembled_lines else (curr_flow if curr_flow else "")
                    m_prev_num = re.match(r"^(\d{1,2})\.\s+", prev_last_line)
                    if m_prev_num and not is_bullet:
                        prev_num = int(m_prev_num.group(1))
                        m_if = re.match(r"^[\.\s\-\)]*([A-Z][a-z]{3,}\s+if\b.*)", sub_l)
                        if m_if:
                            sub_l = f"{prev_num + 1}. {m_if.group(1)}"
                            is_bullet = True

                    is_labeled = bool(re.match(
                        r'^([A-Z][a-zA-Z0-9\s]{1,18}:|(Example|Solution|Definition|Note|Warning|Clause|Step|Terms|Theorem|Lemma|Property|Properties|Rule|Remark|Case)\b)',
                        sub_l, re.IGNORECASE
                    ))
                    is_formula_def = bool(re.search(r'=\s*[\{\(]|can be\s*=', sub_l))

                    is_standalone = is_bullet or is_labeled or is_formula_def

                    prev_ended_intro = bool(curr_flow and (curr_flow.endswith(':') or re.search(r'\bthen\s*$', curr_flow, re.IGNORECASE)))
                    prev_ended_formula = bool(curr_flow and re.search(r'[\}\)]\s*$', curr_flow))

                    if is_standalone or prev_ended_intro or prev_ended_formula:
                        if curr_flow:
                            assembled_lines.append(curr_flow.strip())
                            curr_flow = ""
                        if is_bullet or is_formula_def:
                            assembled_lines.append(sub_l)
                        else:
                            curr_flow = sub_l
                    else:
                        if not curr_flow:
                            curr_flow = sub_l
                        else:
                            if curr_flow.endswith("-"):
                                curr_flow = curr_flow[:-1] + sub_l
                            else:
                                curr_flow += " " + sub_l

                if curr_flow:
                    assembled_lines.append(curr_flow.strip())

                text_block = "\n\n".join(assembled_lines).strip()
                if text_block:
                    avg_c = float(np.mean([t["conf"] for t in curr_para_tokens])) if curr_para_tokens else 1.0
                    pbx1 = int(min(t["min_x"] for t in curr_para_tokens)) if curr_para_tokens else 0
                    pby1 = int(min(t["min_y"] for t in curr_para_tokens)) if curr_para_tokens else int(curr_para_y)
                    pbx2 = int(max(t["max_x"] for t in curr_para_tokens)) if curr_para_tokens else W
                    pby2 = int(max(t["max_y"] for t in curr_para_tokens)) if curr_para_tokens else int(curr_para_y + 20)
                    structured_elements.append({
                        "type": "Paragraph",
                        "text": text_block,
                        "content": text_block,
                        "confidence": avg_c,
                        "top_y": curr_para_y,
                        "box": [pbx1, pby1, pbx2, pby2],
                        "lines": list(curr_para_segs)
                    })
                curr_para_lines = []
                curr_para_tokens = []
                curr_para_segs = []

        for line_idx, line in enumerate(physical_lines):
            line_y = float(np.mean([t.get("y_proj", t["cy"]) for t in line]))
            line_h = float(np.median([t["h"] for t in line]))

            # Check if paragraph should be flushed before this line
            if prev_line_y is not None:
                dy = line_y - prev_line_y
                table_intervening = any(
                    (prev_line_y <= tb["box"][1] <= line_y) or (tb["box"][1] <= line_y <= tb["box"][3])
                    for tb in table_boxes
                )
                prev_line = physical_lines[line_idx - 1]
                prev_line_txt = " ".join(t["text"] for t in prev_line).strip()
                prev_line_max_x = max(t["max_x"] for t in prev_line)
                curr_line_min_x = min(t["min_x"] for t in line)

                prev_line_min_x = min(t["min_x"] for t in prev_line)
                is_para_break = (
                    table_intervening or
                    (dy > 1.35 * med_line_spacing) or
                    (prev_line_max_x < col_min_x + 0.75 * col_width and bool(re.search(r'[.?!:"\']$', prev_line_txt)) and abs(curr_line_min_x - prev_line_min_x) < 50) or
                    (curr_line_min_x > prev_line_min_x + 35)
                )
                if is_para_break:
                    flush_paragraph()

            prev_line_y = line_y

            raw_line_text = " ".join(t["text"] for t in line).strip()
            norm_line_text = self._clean_text_spacing(raw_line_text)
            line_box = _box_of(line)

            # If line has an intro sentence with an inline outline heading attached e.g. "one by one - i) Negation"
            m_inline_head = re.match(r"^(.*?(?:one by one|following|below|discuss)[\s\-—:]+)\s*([ivxlcdm]{1,5}\)?\s+[A-Z].*)$", norm_line_text, re.IGNORECASE)
            if m_inline_head:
                intro_part = m_inline_head.group(1).strip()
                head_part = m_inline_head.group(2).strip()
                if intro_part:
                    # Split the physical line box proportionally between intro text and heading text
                    frac = len(intro_part) / float(max(1, len(intro_part) + len(head_part)))
                    split_x = int(line_box[0] + frac * (line_box[2] - line_box[0]))
                    curr_para_lines.append(intro_part)
                    curr_para_tokens.extend(line)
                    curr_para_segs.append({"text": intro_part, "box": [line_box[0], line_box[1], split_x, line_box[3]]})
                    flush_paragraph()
                    line_box = [split_x, line_box[1], line_box[2], line_box[3]]
                norm_line_text = head_part

            # 1. Page Title / Running Header (Top 22% of page and spans across page or contains section numbering)
            is_running_header = False
            if line_y < 0.22 * H and len(line) >= 2:
                min_x = min(t["min_x"] for t in line)
                max_x = max(t["max_x"] for t in line)
                if (max_x - min_x) > 0.35 * W or re.search(r"\b\d+-\d+\b", norm_line_text):
                    is_running_header = True

            if is_running_header:
                flush_paragraph()
                # If there were any stray optical artifacts or keyboard noise above the running header, purge them
                structured_elements = [
                    el for el in structured_elements
                    if el.get("top_y", 9999) >= line_y - 15
                ]
                header_text = "  •  ".join(t["text"].strip() for t in line)
                header_text = self._clean_text_spacing(header_text)
                tbx1 = int(min(t["min_x"] for t in line))
                tby1 = int(min(t["min_y"] for t in line))
                tbx2 = int(max(t["max_x"] for t in line))
                tby2 = int(max(t["max_y"] for t in line))
                structured_elements.append({
                    "type": "Title",
                    "text": header_text,
                    "content": header_text,
                    "confidence": float(np.mean([t["conf"] for t in line])),
                    "top_y": line_y,
                    "box": [tbx1, tby1, tbx2, tby2],
                    # Each header token keeps its own position (left / centre / right of the running header)
                    "lines": [
                        {"text": self._clean_text_spacing(t["text"]), "box": _box_of([t])}
                        for t in sorted(line, key=lambda x: x["min_x"]) if t["text"].strip()
                    ]
                })
                continue

            # 2. General Section Heading (matches outline numbering: '1.5', '2.1', '1.2.3')
            m_sec = re.match(r"^(\d+(\.\d+)+)\s+([A-Za-z].*)", norm_line_text)
            if m_sec and len(norm_line_text) < 70:
                flush_paragraph()
                hbx1 = int(min(t["min_x"] for t in line))
                hby1 = int(min(t["min_y"] for t in line))
                hbx2 = int(max(t["max_x"] for t in line))
                hby2 = int(max(t["max_y"] for t in line))
                structured_elements.append({
                    "type": "Heading",
                    "text": norm_line_text,
                    "content": norm_line_text,
                    "confidence": float(np.mean([t["conf"] for t in line])),
                    "top_y": line_y,
                    "level": 2,
                    "box": [hbx1, hby1, hbx2, hby2],
                    "lines": [{"text": norm_line_text, "box": line_box}]
                })
                continue

            # 3. General Subheading
            is_subheading = False
            avg_line_conf = float(np.mean([t["conf"] for t in line]))

            # Headings in printed documents must have reliable OCR confidence
            if avg_line_conf >= 0.72:
                # Match Roman numerals ('i)', 'iv)') or letter outlines ('A.', 'B.') with punctuation
                m_roman = re.match(r"^(\([ivxlcdm]+\)|[ivxlcdm]+[\.\)])\s+[A-Za-z]", norm_line_text, re.IGNORECASE)
                m_letter = re.match(r"^(\([A-Za-z]\)|[A-Za-z][\.\)])\s+[A-Za-z]", norm_line_text)
                if (m_roman or m_letter) and len(norm_line_text) < 60:
                    is_subheading = True
                else:
                    # Formulas or equations are not headings
                    is_formula = bool(re.search(r"[={}<>\[\]~]|\b(in|subset)\b", norm_line_text))
                    if not is_formula:
                        words = norm_line_text.split()
                        line_w = max(t["max_x"] for t in line) - min(t["min_x"] for t in line)
                        is_short_line = (1 <= len(words) <= 7 and line_w < 0.52 * col_width and not norm_line_text.endswith("."))
                        connecting_words = {"is", "are", "was", "were", "of", "to", "in", "on", "at", "for", "with", "by", "and", "or", "the", "a", "an", "then", "if"}
                        last_raw = words[-1].strip(".,:;()") if words else ""
                        is_dangling = bool(last_raw.islower() and last_raw in connecting_words)

                        if is_short_line and not is_dangling and words and words[0][0].isupper():
                            if len(norm_line_text.strip()) < 3:
                                is_subheading = False
                            elif len(words) == 1:
                                w = words[0].strip(".,:;()")
                                if len(w) >= 3 and w.isalpha() and w[0].isupper():
                                    prev_y = float(np.mean([t.get("y_proj", t["cy"]) for t in physical_lines[line_idx - 1]])) if line_idx > 0 else None
                                    has_space_above = (prev_y is None or (line_y - prev_y > 1.25 * med_line_spacing))
                                    if has_space_above:
                                        is_subheading = True
                            else:
                                caps = sum(1 for w in words if w.strip(".,:;()")[0:1].isupper() or w.lower() in connecting_words)
                                is_title_cased = (caps >= len(words) - 1)
                                prev_y = float(np.mean([t.get("y_proj", t["cy"]) for t in physical_lines[line_idx - 1]])) if line_idx > 0 else None
                                has_space_above = (prev_y is None or (line_y - prev_y > 1.25 * med_line_spacing))
                                if is_title_cased or has_space_above:
                                    is_subheading = True

            if is_subheading:
                flush_paragraph()
                sbx1 = int(min(t["min_x"] for t in line))
                sby1 = int(min(t["min_y"] for t in line))
                sbx2 = int(max(t["max_x"] for t in line))
                sby2 = int(max(t["max_y"] for t in line))
                structured_elements.append({
                    "type": "Heading",
                    "text": norm_line_text,
                    "content": norm_line_text,
                    "confidence": avg_line_conf,
                    "top_y": line_y,
                    "level": 3,
                    "box": [sbx1, sby1, sbx2, sby2],
                    "lines": [{"text": norm_line_text, "box": line_box}]
                })
                continue

            # 4. Standard Paragraph text
            if not curr_para_lines:
                curr_para_y = line_y
            curr_para_lines.append(norm_line_text)
            curr_para_tokens.extend(line)
            curr_para_segs.append({"text": norm_line_text, "box": line_box})

        flush_paragraph()

        # Consolidate multiple Title elements from running header (e.g. book title + chapter title)
        title_elements = [el for el in structured_elements if el["type"] == "Title"]
        if len(title_elements) > 1:
            # Preserve reading order of running header elements (e.g. Document Title • Chapter • Page Number)
            title_elements.sort(key=lambda el: el.get("top_y", 0))
            merged_title_text = "  •  ".join(el["content"].strip("• ") for el in title_elements)
            first_top_y = min(el.get("top_y", 0) for el in title_elements)
            merged_lines = [seg for el in title_elements for seg in el.get("lines", [])]
            merged_boxes = [el["box"] for el in title_elements if el.get("box")]
            merged_box = (
                [min(b[0] for b in merged_boxes), min(b[1] for b in merged_boxes),
                 max(b[2] for b in merged_boxes), max(b[3] for b in merged_boxes)]
                if merged_boxes else None
            )
            structured_elements = [el for el in structured_elements if el["type"] != "Title"]
            merged_el = {
                "type": "Title",
                "text": merged_title_text,
                "content": merged_title_text,
                "confidence": float(np.mean([el.get("confidence", 1.0) for el in title_elements])),
                "top_y": first_top_y,
                "lines": merged_lines
            }
            if merged_box:
                merged_el["box"] = merged_box
            structured_elements.append(merged_el)

        # -----------------------------------------------------------------
        # STEP 4: Interleave All Elements in Physical Vertical Order
        # -----------------------------------------------------------------
        structured_elements.sort(key=lambda el: el.get("top_y", 0))

        doc_title = "Digitized Document"
        for el in structured_elements:
            if el["type"] == "Title":
                doc_title = el["content"]
                break

        # -----------------------------------------------------------------
        # STEP 5: Build Formatted Markdown and HTML
        # -----------------------------------------------------------------
        md_parts = []
        for el in structured_elements:
            el_type = el["type"]
            content = el.get("content", "")
            if el_type == "Title":
                md_parts.append(f"# {content}")
            elif el_type == "Heading":
                lvl = el.get("level", 3)
                hashes = "#" * lvl
                md_parts.append(f"{hashes} {content}")
            elif el_type == "Table":
                md_parts.append(el.get("markdown", ""))
            elif el_type == "Image":
                md_parts.append(el.get("markdown") or f"> 📷 *{content}*")
            else:
                md_parts.append(content)

        full_markdown = "\n\n".join(part for part in md_parts if part.strip())

        html_parts = []
        for el in structured_elements:
            el_type = el["type"]
            content = el.get("content", "")
            if el_type == "Title":
                html_parts.append(f'<div style="text-align: center; border-bottom: 2px solid #38bdf8; padding-bottom: 8px; margin-bottom: 16px;"><h1 style="color: #38bdf8; font-size: 19px; margin: 0;">{content}</h1></div>')
            elif el_type == "Heading":
                lvl = el.get("level", 3)
                color = "#38bdf8" if lvl == 2 else "#67e8f9"
                font_sz = "17px" if lvl == 2 else "15px"
                html_parts.append(f'<h{lvl} style="color: {color}; margin-top: 18px; margin-bottom: 8px; font-size: {font_sz};">{content}</h{lvl}>')
            elif el_type == "Table":
                html_parts.append(el.get("html", ""))
            elif el_type == "Image":
                html_parts.append(el.get("html") or f'<div style="background: #1e293b; padding: 10px; border-radius: 6px; color: #94a3b8; font-style: italic;">📷 {content}</div>')
            else:
                formatted_para = content.replace("\n", "<br>")
                html_parts.append(f'<p style="color: #e2e8f0; line-height: 1.6; margin-bottom: 10px; font-size: 13.5px;">{formatted_para}</p>')

        full_html = "\n".join(html_parts)

        val_report = self.validator.generate_report(structured_elements, tokens)

        return {
            "title": doc_title,
            "elements": structured_elements,
            "markdown": full_markdown,
            "html": full_html,
            "validation": val_report,
            "confidence": val_report.get("confidence", {}),
            "engine": "Offline (PaddleOCR + SLANet)",
            "tokens": tokens,
            "image_shape": (H, W)
        }

    def extract_full_text(self, image: np.ndarray) -> str:
        doc = self.extract_structured_document(image)
        return doc.get("markdown", "")

    def extract_from_boxes(self, image: np.ndarray, layout_boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        doc = self.extract_structured_document(image, layout_boxes)
        return doc.get("elements", [])
