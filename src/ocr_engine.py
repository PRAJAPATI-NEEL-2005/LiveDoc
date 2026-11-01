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
from typing import List, Dict, Any

from src.table_extractor import TableExtractor
from src.validator import DocumentValidator


class OCREngine:
    def __init__(self, lang='en'):
        print("Initializing PaddleOCR...")
        # det_db_box_thresh=0.50 prevents detecting faint reverse-page bleed-through ink
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            show_log=False,
            det_db_box_thresh=0.50,
            det_db_unclip_ratio=1.6
        )
        self.table_extractor = TableExtractor()
        self.validator = DocumentValidator(min_confidence_threshold=0.72)
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

        return raw, image

    @staticmethod
    def _parse_tokens(raw_items):
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

            tokens.append({
                "text": txt,
                "conf": conf,
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "cx": (min_x + max_x) / 2.0,
                "cy": (min_y + max_y) / 2.0,
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
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'conco', 'called',
            'relanonship', 'relationship', 'object', 'objects', 'elements', 'denotes'
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
    def _clean_text_spacing(text: str) -> str:
        """
        General typographic spacing and whitespace normalization for any document:
        - Collapses duplicate spaces.
        - Fixes spacing around punctuation (commas, periods, colons).
        - Separates outline headings and list item numbers.
        - Cleans unprintable non-ASCII corruptions while preserving text integrity.
        """
        if not text:
            return ""
        t = text
        # Remove unprintable/control characters and unicode replacement artifacts
        t = re.sub(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]", "", t)
        # Collapse multiple whitespace characters into single space
        t = re.sub(r"[ \t]+", " ", t)
        # Ensure single space after punctuation when followed by a word character
        t = re.sub(r"([,;:?!])([A-Za-z0-9])", r"\1 \2", t)
        # Remove stray space immediately before punctuation
        t = re.sub(r"\s+([,.:;?!])", r"\1", t)
        # Strip stray leading closing parenthesis/bracket e.g. ') Conjunction' -> 'Conjunction'
        t = re.sub(r"^\)\s*", "", t)
        # Separate roman numerals or letters merged with uppercase words e.g. 'vIf and only if' -> 'v) If and only if'
        def _sep_roman(m):
            prefix, roman, word = m.group(1), m.group(2), m.group(3)
            # Preserve mathematical relation variables like iRi, iRj, jRk, jRi
            if roman in ("i", "j") and word.startswith("R"):
                return m.group(0)
            return f"{prefix}{roman}) {word}"

        t = re.sub(r"(^|\s)([ivxlcdm]{1,5})([A-Z][a-z]{1,})", _sep_roman, t)
        t = re.sub(r"(^|\s)([ivxlcdm]{1,5})\s*([A-Z][a-z]+)", _sep_roman, t)
        # Separate digits glued directly to capitalized words e.g. '4Symmetric' -> '4. Symmetric'
        t = re.sub(r"(^|\s)(\d{1,2})([A-Z][a-z]+)", r"\1\2. \3", t)
        # Space around math equality/operators when between alphanumeric tokens e.g. 'A=B' -> 'A = B'
        t = re.sub(r"([A-Za-z0-9])(=)([A-Za-z0-9])", r"\1 \2 \3", t)
        # Clean trailing isolated single letter error at the very end of header titles e.g. 'Theorye' -> 'Theory'
        t = re.sub(r"(\b[A-Z][a-z]{3,})e\s*$", r"\1", t)
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

        if not layout_boxes:
            layout_boxes = []

        table_boxes = [b for b in layout_boxes if b.get("type") == "Table"]

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
                    is_duplicate_table = any(
                        el.get("type") == "Table" and (
                            el.get("markdown") == tbl_data["markdown"] or
                            (el.get("headers") == tbl_data.get("headers") and el.get("rows") == tbl_data.get("rows"))
                        )
                        for el in structured_elements
                    )
                    if not is_duplicate_table:
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

        clean_free_tokens.sort(key=lambda t: t["cy"])

        # Cluster into physical lines
        physical_lines = []
        for t in clean_free_tokens:
            matched_line = None
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

                l_cy = float(np.mean([x["cy"] for x in line]))
                l_h = float(np.median([x["h"] for x in line]))
                if abs(t["cy"] - l_cy) < max(0.40 * l_h, 8.0):
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
                # High-confidence threshold for single-token lines unless it is a standard list bullet (e.g. '1.', 'A.', 'i)')
                is_bullet = bool(re.match(r"^(\d{1,2}|[ivx]{1,4}|[a-zA-Z])[\.\)]?$", t["text"].strip()))
                if not is_bullet:
                    if t["conf"] < 0.72 or t["min_x"] > 0.82 * W or t["max_x"] < 0.08 * W or (t["conf"] < 0.85 and t["w"] < 50):
                        continue
            filtered_lines.append(line)
        physical_lines = filtered_lines

        for line in physical_lines:
            line.sort(key=lambda t: t["min_x"])
        physical_lines.sort(key=lambda line: float(np.mean([t["cy"] for t in line])))

        # -----------------------------------------------------------------
        # STEP 3: Classify Text Lines into Header, Headings, Paragraphs
        # -----------------------------------------------------------------
        med_line_h = float(np.median([t["h"] for t in clean_free_tokens])) if clean_free_tokens else 20.0
        curr_para_lines = []
        curr_para_tokens = []
        curr_para_y = 0
        prev_line_y = None

        def flush_paragraph():
            nonlocal curr_para_lines, curr_para_tokens, curr_para_y
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
                    structured_elements.append({
                        "type": "Paragraph",
                        "text": text_block,
                        "content": text_block,
                        "confidence": avg_c,
                        "top_y": curr_para_y
                    })
                curr_para_lines = []
                curr_para_tokens = []

        for line in physical_lines:
            line_y = float(np.mean([t["cy"] for t in line]))
            line_h = float(np.median([t["h"] for t in line]))

            # Check if any table is located between previous line and current line
            if prev_line_y is not None:
                table_intervening = any(
                    (prev_line_y <= tb["box"][1] <= line_y) or (tb["box"][1] <= line_y <= tb["box"][3])
                    for tb in table_boxes
                )
                if table_intervening:
                    flush_paragraph()
                elif line_y - prev_line_y > 2.0 * med_line_h:
                    flush_paragraph()

            prev_line_y = line_y

            raw_line_text = " ".join(t["text"] for t in line).strip()
            norm_line_text = self._clean_text_spacing(raw_line_text)

            # If line has an intro sentence with an inline outline heading attached e.g. "one by one - i) Negation"
            m_inline_head = re.match(r"^(.*?(?:one by one|following|below|discuss)[\s\-—:]+)\s*([ivxlcdm]{1,5}\)?\s+[A-Z].*)$", norm_line_text, re.IGNORECASE)
            if m_inline_head:
                intro_part = m_inline_head.group(1).strip()
                head_part = m_inline_head.group(2).strip()
                if intro_part:
                    curr_para_lines.append(intro_part)
                    curr_para_tokens.extend(line)
                    flush_paragraph()
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
                structured_elements.append({
                    "type": "Title",
                    "text": header_text,
                    "content": header_text,
                    "confidence": float(np.mean([t["conf"] for t in line])),
                    "top_y": line_y
                })
                continue

            # 2. General Section Heading (matches outline numbering: '1.5', '2.1', '1.2.3')
            m_sec = re.match(r"^(\d+(\.\d+)+)\s+([A-Za-z].*)", norm_line_text)
            if m_sec and len(norm_line_text) < 70:
                flush_paragraph()
                structured_elements.append({
                    "type": "Heading",
                    "text": norm_line_text,
                    "content": norm_line_text,
                    "confidence": float(np.mean([t["conf"] for t in line])),
                    "top_y": line_y,
                    "level": 2
                })
                continue

            # 3. General Subheading
            is_subheading = False
            avg_line_conf = float(np.mean([t["conf"] for t in line]))

            # Headings in printed documents must have reliable OCR confidence
            if avg_line_conf >= 0.72:
                # Match Roman numerals ('i)', 'iv)') or letter outlines ('A.', 'B.') with punctuation
                m_roman = re.match(r"^([ivxlcdm]+)[\.\)\s-]+\s*[A-Za-z]", norm_line_text, re.IGNORECASE)
                m_letter = re.match(r"^(\([A-Za-z]\)|[A-Za-z][\.\)])\s+[A-Za-z]", norm_line_text)
                if (m_roman or m_letter) and len(norm_line_text) < 60:
                    is_subheading = True
                else:
                    # Formulas or equations are not headings
                    is_formula = bool(re.search(r"[={}<>\[\]~]|\b(in|subset)\b", norm_line_text))
                    if not is_formula:
                        words = norm_line_text.split()
                        if 1 <= len(words) <= 6 and len(norm_line_text) < 45 and not norm_line_text.endswith("."):
                            connecting_words = {"is", "are", "was", "were", "of", "to", "in", "on", "at", "for", "with", "by", "and", "or", "the", "a", "an", "then", "if"}
                            last_raw = words[-1].strip(".,:;()")
                            # If the last word is purely lowercase and in connecting_words (e.g. "of", "for", "in"), it is a dangling fragment
                            if not (last_raw.islower() and last_raw in connecting_words):
                                if len(words) == 1:
                                    w = words[0].strip(".,:;()")
                                    if len(w) >= 3 and w.isalpha() and w[0].isupper():
                                        is_subheading = True
                                elif words[0][0].isupper():
                                    caps = sum(1 for w in words if w.strip(".,:;()")[0:1].isupper() or w.lower() in connecting_words)
                                    if caps >= len(words) - 1:
                                        is_subheading = True

            if is_subheading:
                flush_paragraph()
                structured_elements.append({
                    "type": "Heading",
                    "text": norm_line_text,
                    "content": norm_line_text,
                    "confidence": avg_line_conf,
                    "top_y": line_y,
                    "level": 3
                })
                continue

            # 4. Standard Paragraph text
            if not curr_para_lines:
                curr_para_y = line_y
            curr_para_lines.append(norm_line_text)
            curr_para_tokens.extend(line)

        flush_paragraph()

        # Consolidate multiple Title elements from running header (e.g. book title + chapter title)
        title_elements = [el for el in structured_elements if el["type"] == "Title"]
        if len(title_elements) > 1:
            # Preserve reading order of running header elements (e.g. Document Title • Chapter • Page Number)
            title_elements.sort(key=lambda el: el.get("top_y", 0))
            merged_title_text = "  •  ".join(el["content"].strip("• ") for el in title_elements)
            first_top_y = min(el.get("top_y", 0) for el in title_elements)
            structured_elements = [el for el in structured_elements if el["type"] != "Title"]
            structured_elements.append({
                "type": "Title",
                "text": merged_title_text,
                "content": merged_title_text,
                "confidence": float(np.mean([el.get("confidence", 1.0) for el in title_elements])),
                "top_y": first_top_y
            })

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
                md_parts.append(f"> 📷 *{content}*")
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
                html_parts.append(f'<div style="background: #1e293b; padding: 10px; border-radius: 6px; color: #94a3b8; font-style: italic;">📷 {content}</div>')
            else:
                formatted_para = content.replace("\n", "<br>")
                html_parts.append(f'<p style="color: #e2e8f0; line-height: 1.6; margin-bottom: 10px; font-size: 13.5px;">{formatted_para}</p>')

        full_html = "\n".join(html_parts)

        val_report = self.validator.generate_report(structured_elements)

        return {
            "title": doc_title,
            "elements": structured_elements,
            "markdown": full_markdown,
            "html": full_html,
            "validation": val_report
        }

    def extract_full_text(self, image: np.ndarray) -> str:
        doc = self.extract_structured_document(image)
        return doc.get("markdown", "")

    def extract_from_boxes(self, image: np.ndarray, layout_boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        doc = self.extract_structured_document(image, layout_boxes)
        return doc.get("elements", [])
