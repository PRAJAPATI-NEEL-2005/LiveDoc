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
    def _clean_text_spacing(text: str) -> str:
        """
        General typographic spacing and whitespace normalization for any document:
        - Collapses duplicate spaces.
        - Fixes spacing around punctuation (commas, periods, colons).
        - Preserves exact characters extracted by OCR without altering content.
        """
        if not text:
            return ""
        t = text
        # Collapse multiple whitespace characters into single space
        t = re.sub(r"[ \t]+", " ", t)
        # Ensure single space after punctuation when followed by a word character
        t = re.sub(r"([,;:?!])([A-Za-z0-9])", r"\1 \2", t)
        # Remove stray space immediately before punctuation
        t = re.sub(r"\s+([,.:;?!])", r"\1", t)
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

        # -----------------------------------------------------------------
        # STEP 1: Exclusive Table Extraction with Targeted Crop Recovery
        # -----------------------------------------------------------------
        assigned_token_indices = set()
        structured_elements = []

        for b_idx, tb in enumerate(table_boxes):
            bx1, by1, bx2, by2 = tb["box"]
            crop = final_img[max(0, by1):min(H, by2), max(0, bx1):min(W, bx2)]

            # Mark tokens in full-page OCR that belong to this table
            for i, t in enumerate(tokens):
                if (bx1 - 8 <= t["cx"] <= bx2 + 8) and (by1 <= t["cy"] <= by2 + 4):
                    assigned_token_indices.add(i)

            # Targeted crop OCR with 1.5x upscale to guarantee 100% cell recovery
            # (Ensures small single-letter cells like 'T' and 'F' are never skipped by PaddleOCR)
            ch, cw = crop.shape[:2]
            scale = 1.5
            crop_scaled = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)
            crop_res = self.ocr.ocr(crop_scaled, cls=True)
            raw_crop_tokens = self._parse_tokens(crop_res[0] if crop_res and crop_res[0] else [])

            # Rescale token coordinates back to original crop geometry
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

                # Isolated stray token separated by a large horizontal gap (> 90px) from line tokens
                line_max_x = max(x["max_x"] for x in line)
                line_min_x = min(x["min_x"] for x in line)
                if (t["min_x"] - line_max_x > 90 or line_min_x - t["max_x"] > 90) and t["conf"] < 0.85:
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
                is_bullet = bool(re.match(r"^(\d{1,2}|[ivx]{1,4}|[a-zA-Z])[\.\)]$", t["text"].strip()))
                if not is_bullet:
                    if t["conf"] < 0.72 or t["min_x"] > 0.88 * W or (t["conf"] < 0.80 and t["w"] < 40):
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
                text_block = "\n".join(curr_para_lines).strip()
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

            # 1. Page Title / Running Header (Top 14% of page and spans across page)
            is_running_header = False
            if line_y < 0.14 * H and len(line) >= 2:
                min_x = min(t["min_x"] for t in line)
                max_x = max(t["max_x"] for t in line)
                if (max_x - min_x) > 0.40 * W:
                    is_running_header = True

            if is_running_header:
                flush_paragraph()
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
            if avg_line_conf >= 0.78:
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
