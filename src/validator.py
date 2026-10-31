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

    def generate_report(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates a comprehensive validation report covering all document elements.
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

        return {
            "status": status,
            "total_elements": len(elements),
            "uncertain_items_count": len(uncertain_items),
            "inconsistencies_count": len(inconsistencies),
            "uncertain_items": uncertain_items,
            "inconsistencies": inconsistencies
        }
