"""
LiveDoc Validation and Verification Engine
Implements the validation layer defined in the project architecture:
- Distinguishes between OCR uncertainty and actual document inconsistencies.
- Principles: Extract -> Validate -> Flag -> Preserve Original -> Report.
- Checks OCR confidence scores.
- Checks arithmetic consistency (e.g., Table Unit Price * Qty == Total).
- Validates common structured formats (Dates, Numbers, Invoices).
"""

import re
from typing import List, Dict, Any


class DocumentValidator:
    def __init__(self, min_confidence_threshold: float = 0.72):
        self.min_confidence = min_confidence_threshold

    def validate_confidence(self, text: str, confidence: float) -> Dict[str, Any]:
        """
        Flag text that has lower confidence without silently modifying it.
        """
        is_uncertain = confidence < self.min_confidence
        return {
            "text": text,
            "confidence": round(confidence, 3),
            "uncertain": is_uncertain,
            "status": "Verification Required" if is_uncertain else "Valid"
        }

    def validate_table_consistency(self, table_element: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Inspect numeric rows in a table to verify mathematical consistency
        (e.g., Quantity * Unit Price = Total, or sum of items = Grand Total).
        Does NOT alter printed values; flags any discrepancies found.
        """
        flags = []
        rows = table_element.get("rows", [])
        headers = [h.strip().lower() for h in table_element.get("headers", [])]

        if not rows:
            return flags

        # Search for columns related to Qty, Price, Amount/Total
        qty_col = -1
        price_col = -1
        total_col = -1

        for idx, h in enumerate(headers):
            if any(term in h for term in ["qty", "quantity", "count", "units"]):
                qty_col = idx
            elif any(term in h for term in ["price", "rate", "cost", "unit"]):
                price_col = idx
            elif any(term in h for term in ["total", "amount", "subtotal", "val"]):
                total_col = idx

        # 1. Row-level arithmetic check: Qty * Price == Total
        if qty_col != -1 and price_col != -1 and total_col != -1:
            for r_idx, row in enumerate(rows):
                if len(row) <= max(qty_col, price_col, total_col):
                    continue
                try:
                    q_str = re.sub(r"[^\d.]", "", str(row[qty_col]))
                    p_str = re.sub(r"[^\d.]", "", str(row[price_col]))
                    t_str = re.sub(r"[^\d.]", "", str(row[total_col]))

                    if q_str and p_str and t_str:
                        qty = float(q_str)
                        price = float(p_str)
                        printed_total = float(t_str)
                        calculated = round(qty * price, 2)

                        if abs(calculated - printed_total) > 0.05:
                            flags.append({
                                "type": "arithmetic_inconsistency",
                                "row": r_idx + 1,
                                "message": f"Row {r_idx + 1}: Printed total ({printed_total}) differs from Qty x Price ({calculated})",
                                "printed_value": printed_total,
                                "calculated_value": calculated,
                                "status": "Inconsistent"
                            })
                except (ValueError, TypeError):
                    continue

        # 2. Grand total column summation check
        if total_col != -1 and len(rows) >= 3:
            # Check if last row is a 'Total' row
            last_row = rows[-1]
            first_cell = str(last_row[0]).strip().lower() if last_row else ""
            if any(term in first_cell for term in ["total", "sum", "grand"]):
                try:
                    t_str = re.sub(r"[^\d.]", "", str(last_row[total_col]))
                    if t_str:
                        printed_sum = float(t_str)
                        calc_sum = 0.0
                        valid_items = 0
                        for r in rows[:-1]:
                            if len(r) > total_col:
                                val_str = re.sub(r"[^\d.]", "", str(r[total_col]))
                                if val_str:
                                    calc_sum += float(val_str)
                                    valid_items += 1

                        calc_sum = round(calc_sum, 2)
                        if valid_items >= 2 and abs(calc_sum - printed_sum) > 0.05:
                            flags.append({
                                "type": "column_sum_inconsistency",
                                "message": f"Printed Grand Total ({printed_sum}) differs from sum of line items ({calc_sum})",
                                "printed_value": printed_sum,
                                "calculated_value": calc_sum,
                                "status": "Inconsistent"
                            })
                except (ValueError, TypeError):
                    pass

        return flags

    def compute_confidence_score(self, elements: List[Dict[str, Any]], tokens: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Computes a rigorous, multi-factor confidence score S in [0, 100] for the scanned document:
        S = 0.45 * C_ocr + 0.30 * C_lex + 0.15 * C_table + 0.10 * C_layout
        Returns detailed sub-scores, qualitative grade, and recommendation for Smart AI auto-fallback.
        """
        # 1. OCR Token Confidence (C_ocr)
        if tokens:
            confs = [float(t.get("conf", 1.0)) for t in tokens if "conf" in t]
            if confs:
                avg_conf = (sum(confs) / len(confs)) * 100.0
                low_conf_ratio = sum(1 for c in confs if c < self.min_confidence) / len(confs)
                penalty = min(30.0, low_conf_ratio * 50.0)
                c_ocr = max(0.0, min(100.0, avg_conf - penalty))
            else:
                c_ocr = 90.0
        else:
            el_confs = [float(el.get("confidence", 1.0)) for el in elements if "confidence" in el]
            c_ocr = (sum(el_confs) / max(1, len(el_confs))) * 100.0 if el_confs else 90.0

        # 2. Lexical / Word Integrity (C_lex)
        all_words = []
        for el in elements:
            txt = el.get("content", "") or el.get("text", "")
            if txt:
                words = re.findall(r"\b[A-Za-z]{2,}\b", txt)
                all_words.extend(words)

        if all_words:
            valid_words = 0
            for w in all_words:
                lw = w.lower()
                # Check for common OCR character corruptions:
                # - Impossible consonant clusters in English (e.g. 5+ consonants in a row)
                has_impossible_consonants = bool(re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", lw))
                # - Words missing vowels (except legitimate abbreviations/roman numerals)
                has_no_vowels = not bool(re.search(r"[aeiouy]", lw)) and len(lw) > 3
                # - Non-alphabetic internal noise
                if not has_impossible_consonants and not has_no_vowels:
                    valid_words += 1
            c_lex = (valid_words / float(len(all_words))) * 100.0
        else:
            c_lex = 92.0

        # 3. Table Structural Integrity (C_table)
        tables = [el for el in elements if el.get("type") == "Table"]
        needs_escalation = False
        if tables:
            tbl_penalties = 0.0
            for tbl in tables:
                headers = tbl.get("headers", [])
                rows = tbl.get("rows", [])
                all_cells = [str(h) for h in headers] + [str(c) for r in rows for c in r]
                total_cells = len(all_cells)
                if total_cells > 0:
                    empty_cells = sum(1 for c in all_cells if not c.strip())
                    empty_ratio = empty_cells / float(total_cells)
                    if empty_ratio > 0.15:
                        tbl_penalties += empty_ratio * 50.0

                    # Check if cells contain narrative sentences or multi-word prose (false table detection)
                    # Genuine data tables have compact cell values (typically numbers, codes, dates, short names).
                    # A false table (where prose paragraphs or page frames were mistakenly boxed) contains long text in cells.
                    long_cells = sum(1 for c in all_cells if len(c.split()) >= 4 or len(c) > 30)
                    if (long_cells / float(total_cells)) > 0.25:
                        tbl_penalties += 45.0
                        needs_escalation = True

                # Check arithmetic flags
                arith_flags = self.validate_table_consistency(tbl)
                tbl_penalties += len(arith_flags) * 15.0

            c_table = max(20.0, min(100.0, 100.0 - tbl_penalties))
        else:
            c_table = 100.0

        # 4. Document Layout Coherence (C_layout)
        has_title = any(el.get("type") == "Title" for el in elements)
        has_content = len(elements) > 0
        c_layout = 85.0
        if has_title:
            c_layout += 10.0
        if has_content:
            c_layout += 5.0

        # Check for fragmented snippet noise (e.g. isolated numbers or 1-2 word lines from circular graphics)
        short_elements = [
            el for el in elements 
            if el.get("type") in ("Paragraph", "Heading") 
            and 0 < len(str(el.get("content", "")).split()) <= 2
            and str(el.get("content", "")).strip() not in ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.")
        ]
        if len(elements) >= 8 and (len(short_elements) / float(len(elements))) > 0.20:
            c_layout = max(35.0, c_layout - 35.0)
            needs_escalation = True

        c_layout = min(100.0, c_layout)

        # 5. Composite Final Score
        composite_score = (0.45 * c_ocr) + (0.30 * c_lex) + (0.15 * c_table) + (0.10 * c_layout)
        composite_score = round(max(0.0, min(100.0, composite_score)), 1)

        # Determine qualitative grade and escalation requirement
        if composite_score >= 90.0:
            grade = "EXCELLENT"
        elif composite_score >= 84.0:
            grade = "GOOD"
        elif composite_score >= 70.0:
            grade = "FAIR"
            needs_escalation = True
        else:
            grade = "LOW"
            needs_escalation = True

        # Escalate immediately if critical table arithmetic inconsistency was detected
        if tables:
            for tbl in tables:
                if self.validate_table_consistency(tbl):
                    needs_escalation = True
                    break

        # Escalate if multiple images or visual graphics are present
        image_count = sum(1 for el in elements if el.get("type") == "Image")
        if image_count >= 2:
            needs_escalation = True

        # Escalate if text explicitly references a Figure/Diagram but no Image element was extracted
        has_image = any(el.get("type") == "Image" for el in elements)
        all_text = " ".join([str(el.get("content", "")) for el in elements])
        if re.search(r"(?i)\b(?:fig|figure|diagram)\b(?:\s*[\.\d]+)?", all_text) and not has_image:
            needs_escalation = True

        # Escalate if mathematical, algebraic, or formal symbolic syntax is present
        # Genuine math contains operators, equations, brackets, Greek letters, or relational notation
        has_symbolic_math = bool(re.search(
            r"[\u0370-\u03ff\u2200-\u22ff]"                           # Greek letters & math symbols
            r"|\\(?:[a-zA-Z]+)"                                       # LaTeX formatting (\to, \land, etc.)
            r"|(?:\b[A-Za-z]\s*=\s*[\{\(\[])"                         # Set/tuple definition e.g. A = { ... }, R = ( ... )
            r"|(?:\b[A-Za-z]\s*(?:->|=>|↔|→|⇒)\s*[A-Za-z]\b)"          # Function mapping or implication arrows
            r"|(?:\b(?:[a-zA-Z0-9_]+\s*[\+\-\*\/\^=]\s*){2,})"        # Multi-term algebraic formulas e.g. a + b = c
            r"|(?:\b(?:sin|cos|tan|log|lim|max|min|det|mod)\b\s*[\(\{\[])", # Standard mathematical functions
            all_text, re.IGNORECASE
        ))
        if has_symbolic_math:
            needs_escalation = True

        # Escalate if boolean truth table or binary matrix structure is detected
        if tables:
            for tbl in tables:
                rows = tbl.get("rows", [])
                cell_vals = [str(c).strip().upper() for r in rows for c in r if str(c).strip()]
                if len(cell_vals) >= 4:
                    bool_count = sum(1 for c in cell_vals if c in ("T", "F", "0", "1", "TRUE", "FALSE"))
                    if (bool_count / float(len(cell_vals))) >= 0.50:
                        needs_escalation = True
                        break

        # Escalate if table or layout structural confidence dropped significantly
        if c_table <= 60.0 or c_layout <= 65.0:
            needs_escalation = True

        return {
            "score": composite_score,
            "grade": grade,
            "needs_escalation": needs_escalation,
            "ocr_confidence": round(c_ocr, 1),
            "lexical_confidence": round(c_lex, 1),
            "table_confidence": round(c_table, 1),
            "layout_confidence": round(c_layout, 1),
            "summary": f"{grade} Confidence ({composite_score}%)"
        }

    def generate_report(self, elements: List[Dict[str, Any]], tokens: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generates a comprehensive validation report covering all document elements
        along with the composite confidence score.
        """
        uncertain_items = []
        inconsistencies = []

        for idx, el in enumerate(elements):
            el_type = el.get("type", "Unknown")
            conf = el.get("confidence", 1.0)
            text = el.get("content", "")

            # Check OCR confidence
            if conf < self.min_confidence and text:
                uncertain_items.append({
                    "element_index": idx + 1,
                    "type": el_type,
                    "text": text[:60] + "..." if len(text) > 60 else text,
                    "confidence": round(conf, 3),
                    "flag": "Low OCR Confidence"
                })

            # Check table data consistency
            if el_type == "Table":
                tbl_flags = self.validate_table_consistency(el)
                for f in tbl_flags:
                    f["element_index"] = idx + 1
                    inconsistencies.append(f)

        status = "NEEDS_VERIFICATION" if (uncertain_items or inconsistencies) else "VERIFIED"
        confidence_report = self.compute_confidence_score(elements, tokens)

        return {
            "status": status,
            "total_elements": len(elements),
            "uncertain_items_count": len(uncertain_items),
            "inconsistencies_count": len(inconsistencies),
            "uncertain_items": uncertain_items,
            "inconsistencies": inconsistencies,
            "confidence": confidence_report
        }
