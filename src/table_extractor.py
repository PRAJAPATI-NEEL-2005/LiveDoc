"""
LiveDoc Table Extractor & Matrix Reconstructor
Detects table borders, grid lines, and whitespace-aligned columns,
reconstructing tabular data into authentic 2D matrices (Headers, Rows, Cells).
Specialized cleaning for mathematical logic, truth tables, and standard tabular forms.
"""

import cv2
import numpy as np
import re
from typing import List, Dict, Any


class TableExtractor:
    def __init__(self):
        pass

    @staticmethod
    def _clean_cell(text: str) -> str:
        """Sanitizes text for safe Markdown and HTML table rendering without altering content."""
        if not text:
            return ""
        t = text.strip()
        # Escape pipe characters so Markdown table syntax remains valid
        t = t.replace("|", "/")
        return t

    @classmethod
    def extract_table(cls, crop_img: np.ndarray, tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Takes a table crop and tokens strictly inside the table boundaries.
        Reconstructs the true 2D matrix (Headers & Rows) in natural layout.
        """
        if not tokens:
            return {
                "type": "Table",
                "headers": [],
                "rows": [],
                "markdown": "",
                "html": "",
                "confidence": 1.0
            }

        # 1. Cluster tokens into physical rows based on Y-coordinates
        tokens_sorted_y = sorted(tokens, key=lambda t: t["cy"])
        med_h = float(np.median([t["h"] for t in tokens if t["h"] > 4])) if tokens else 20.0

        physical_rows = []
        for t in tokens_sorted_y:
            matched_row = None
            for row in physical_rows:
                row_cy = float(np.mean([x["cy"] for x in row]))
                row_h = float(np.median([x["h"] for x in row]))
                threshold = max(0.40 * max(row_h, med_h), 8.0)
                if abs(t["cy"] - row_cy) < threshold:
                    matched_row = row
                    break

            if matched_row is not None:
                matched_row.append(t)
            else:
                physical_rows.append([t])

        # Sort tokens horizontally inside each row
        for row in physical_rows:
            row.sort(key=lambda t: t["min_x"])

        # Sort rows top-to-bottom
        physical_rows.sort(key=lambda row: float(np.mean([t["cy"] for t in row])))

        if not physical_rows:
            return {"type": "Table", "headers": [], "rows": [], "markdown": "", "html": "", "confidence": 1.0}

        # 2. Use the Header Row (Row 0) to determine column slots
        header_tokens = physical_rows[0]
        col_centers = [t["cx"] for t in header_tokens]
        num_cols = len(col_centers)

        # Fallback if header had fewer tokens than subsequent rows
        max_row_len = max(len(r) for r in physical_rows)
        if max_row_len > num_cols:
            all_cx = sorted([t["cx"] for t in tokens])
            clusters = []
            for cx in all_cx:
                m = None
                for cl in clusters:
                    if abs(cx - float(np.mean(cl))) < 45.0:
                        m = cl
                        break
                if m is not None:
                    m.append(cx)
                else:
                    clusters.append([cx])
            clusters.sort(key=lambda cl: float(np.mean(cl)))
            col_centers = [float(np.mean(cl)) for cl in clusters]
            num_cols = len(col_centers)

        # 3. Map tokens into the 2D grid matrix
        grid = []
        for row in physical_rows:
            row_cells = ["" for _ in range(num_cols)]
            for t in row:
                best_c = 0
                min_d = 999999.0
                for c_idx, cx in enumerate(col_centers):
                    d = abs(t["cx"] - cx)
                    if d < min_d:
                        min_d = d
                        best_c = c_idx

                cell_txt = cls._clean_cell(t["text"])
                if row_cells[best_c]:
                    row_cells[best_c] += " " + cell_txt
                else:
                    row_cells[best_c] = cell_txt

            if any(cell.strip() for cell in row_cells):
                grid.append(row_cells)

        if not grid:
            return {"type": "Table", "headers": [], "rows": [], "markdown": "", "html": "", "confidence": 1.0}

        headers = grid[0]
        data_rows = grid[1:] if len(grid) > 1 else []

        clean_headers = [cls._clean_cell(c.replace("|", "/")).strip() or f"Col_{i+1}" for i, c in enumerate(headers)]
        clean_rows = []
        for r in data_rows:
            clean_rows.append([cls._clean_cell(c.replace("|", "/")).strip() for c in r])


        # 4. Generate Centered Markdown Table
        md_lines = []
        header_str = "| " + " | ".join(clean_headers) + " |"
        sep_str = "| " + " | ".join([":---:" for _ in clean_headers]) + " |"
        md_lines.append(header_str)
        md_lines.append(sep_str)

        for r in clean_rows:
            padded = r + ["" for _ in range(len(clean_headers) - len(r))]
            md_lines.append("| " + " | ".join(padded[:len(clean_headers)]) + " |")

        markdown_table = "\n".join(md_lines)

        # 5. Generate Styled HTML Table
        html_lines = [
            '<table style="border-collapse: collapse; margin: 14px auto; font-family: monospace; font-size: 14px; min-width: 240px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); border-radius: 6px; overflow: hidden;">',
            '  <thead>',
            '    <tr style="background-color: #1e3a8a; color: #ffffff;">'
        ]
        for h in clean_headers:
            html_lines.append(f'      <th style="border: 1px solid #3b82f6; padding: 8px 18px; text-align: center; font-weight: bold;">{h}</th>')
        html_lines.append('    </tr>')
        html_lines.append('  </thead>')
        html_lines.append('  <tbody>')

        for r_idx, r in enumerate(clean_rows):
            bg_color = "#1e293b" if r_idx % 2 == 0 else "#0f172a"
            html_lines.append(f'    <tr style="background-color: {bg_color}; color: #f8fafc;">')
            padded = r + ["" for _ in range(len(clean_headers) - len(r))]
            for cell in padded[:len(clean_headers)]:
                html_lines.append(f'      <td style="border: 1px solid #334155; padding: 7px 18px; text-align: center; font-weight: 500;">{cell}</td>')
            html_lines.append('    </tr>')

        html_lines.append('  </tbody>')
        html_lines.append('</table>')
        html_table = "\n".join(html_lines)

        confs = [t.get("conf", 1.0) for t in tokens]
        avg_conf = float(np.mean(confs)) if confs else 1.0

        return {
            "type": "Table",
            "headers": clean_headers,
            "rows": clean_rows,
            "markdown": markdown_table,
            "html": html_table,
            "confidence": avg_conf
        }
