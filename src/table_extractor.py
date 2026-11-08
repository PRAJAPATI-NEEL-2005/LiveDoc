"""
LiveDoc Table Extractor & Matrix Reconstructor
Detects table borders, grid lines, and whitespace-aligned columns,
reconstructing tabular data into authentic 2D matrices (Headers, Rows, Cells).
Specialized matrix reconstruction for numeric data, grids, and standard tabular forms.
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
        # Escape pipe characters so Markdown table syntax remains valid without ugly slashes
        t = t.replace("|", "│")
        # Strip trailing isolated footnote/page digits separated by space from short logic headers (e.g. "A^B 10" -> "A^B")
        t = re.sub(r"^([A-Za-z\~\^\v\-]{1,6})\s+\d{1,2}$", r"\1", t)
        return t

    @classmethod
    def extract_table(cls, crop_img: np.ndarray, tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Takes a table crop and tokens strictly inside the table boundaries.
        Reconstructs the true 2D matrix (Headers & Rows) in natural layout
        using multi-row column consensus voting to eliminate margin outliers.
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

        # Pre-process tokens: split multi-cell tokens grouped by OCR (e.g. "F F", "T T", "0 1")
        expanded_tokens = []
        for t in tokens:
            parts = t["text"].strip().split()
            if len(parts) >= 2 and all(len(p) <= 4 for p in parts) and t["w"] > 22:
                part_w = t["w"] / float(len(parts))
                for p_idx, p in enumerate(parts):
                    px1 = t["min_x"] + p_idx * part_w
                    px2 = px1 + part_w
                    expanded_tokens.append({
                        "text": p,
                        "conf": t.get("conf", 1.0),
                        "min_x": px1,
                        "max_x": px2,
                        "min_y": t["min_y"],
                        "max_y": t["max_y"],
                        "cx": (px1 + px2) / 2.0,
                        "cy": t["cy"],
                        "h": t["h"],
                        "w": part_w
                    })
            else:
                expanded_tokens.append(t)
        tokens = expanded_tokens

        # 1. Cluster tokens into physical rows based on Y-coordinates
        tokens_sorted_y = sorted(tokens, key=lambda t: t["cy"])
        med_h = float(np.median([t["h"] for t in tokens if t["h"] > 4])) if tokens else 20.0
        med_w = float(np.median([t["w"] for t in tokens if t["w"] > 4])) if tokens else 25.0

        physical_rows = []
        for t in tokens_sorted_y:
            matched_row = None
            for row in physical_rows:
                row_cy = float(np.mean([x["cy"] for x in row]))
                row_h = float(np.median([x["h"] for x in row]))
                threshold = max(0.55 * max(row_h, med_h), 11.0)
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

        num_physical_rows = len(physical_rows)

        # 2. Multi-Row Column Consensus (Density-based X clustering across ALL rows)
        all_cx = sorted([t["cx"] for t in tokens])
        clusters = []
        table_w = max(t["max_x"] for t in tokens) - min(t["min_x"] for t in tokens)
        thresh_x = max(med_w * 0.85, min(table_w / 7.0, 50.0))

        for cx in all_cx:
            matched_cluster = None
            for cl in clusters:
                if abs(cx - float(np.mean(cl))) < thresh_x:
                    matched_cluster = cl
                    break
            if matched_cluster is not None:
                matched_cluster.append(cx)
            else:
                clusters.append([cx])

        # Purge isolated outlier columns (e.g. margin text on the far edge that only appears in 1 row)
        if num_physical_rows >= 3:
            min_col_tokens = max(2, int(np.ceil(num_physical_rows * 0.40)))
            valid_clusters = [cl for cl in clusters if len(cl) >= min_col_tokens]
        else:
            valid_clusters = clusters

        if not valid_clusters:
            valid_clusters = clusters

        valid_clusters.sort(key=lambda cl: float(np.mean(cl)))
        col_centers = [float(np.mean(cl)) for cl in valid_clusters]
        num_cols = len(col_centers)

        # A table must have at least 2 columns and at least 4 tokens across at least 2 rows
        if num_cols < 2 or len(tokens) < 4 or len(physical_rows) < 2:
            return {"type": "Table", "headers": [], "rows": [], "markdown": "", "html": "", "confidence": 1.0}

        # 3. Map tokens into the 2D grid matrix
        grid = []
        for row in physical_rows:
            row_cells = ["" for _ in range(num_cols)]
            for t in sorted(row, key=lambda x: x["min_x"]):
                # Find closest consensus column center
                best_c = min(range(num_cols), key=lambda c_idx: abs(t["cx"] - col_centers[c_idx]))

                cell_txt = cls._clean_cell(t["text"])
                if row_cells[best_c]:
                    row_cells[best_c] += " " + cell_txt
                else:
                    row_cells[best_c] = cell_txt

            if any(cell.strip() for cell in row_cells):
                grid.append(row_cells)

        if not grid:
            return {"type": "Table", "headers": [], "rows": [], "markdown": "", "html": "", "confidence": 1.0}

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

        # Filter out stray rows that consist only of English prose stopwords
        clean_grid = []
        for row_cells in grid:
            non_empty = [c.strip() for c in row_cells if c.strip()]
            if not non_empty:
                continue
            is_stray = all(any(p.lower() in STOPWORDS for p in c.split()) for c in non_empty)
            if not is_stray:
                clean_grid.append(row_cells)
        if clean_grid:
            grid = clean_grid

        headers = grid[0]
        data_rows = grid[1:] if len(grid) > 1 else []

        # Clean headers: preserve authentic header text without injecting artificial Col_X tokens
        clean_headers = [cls._clean_cell(c.replace("|", "│")).strip() for c in headers]
        # If any header cell is empty, fill with standard non-disruptive label or space
        clean_headers = [c if c else " " for c in clean_headers]

        clean_rows = []
        for r in data_rows:
            clean_rows.append([cls._clean_cell(c.replace("|", "│")).strip() for c in r])

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

        # 5. Generate Publication-Styled HTML Table
        is_compact = (len(clean_headers) <= 4 and all(len(c) <= 6 for c in clean_headers))
        tbl_width = "width: auto; max-width: 280px;" if is_compact else "width: 85%; max-width: 540px;"

        html_lines = [
            f'<div style="page-break-inside: avoid; break-inside: avoid; margin: 12px 0;">',
            f'  <table style="border-collapse: collapse; margin: 0 auto; font-family: monospace; font-size: 13.5px; {tbl_width} box-shadow: 0 2px 4px rgba(0,0,0,0.15); border-radius: 4px; overflow: hidden; page-break-inside: avoid;">',
            '    <thead>',
            '      <tr style="background-color: #1e3a8a; color: #ffffff;">'
        ]
        for h in clean_headers:
            html_lines.append(f'        <th style="border: 1px solid #3b82f6; padding: 7px 14px; text-align: center; font-weight: bold;">{h}</th>')
        html_lines.append('      </tr>')
        html_lines.append('    </thead>')
        html_lines.append('    <tbody>')

        for r_idx, r in enumerate(clean_rows):
            bg_color = "#1e293b" if r_idx % 2 == 0 else "#0f172a"
            html_lines.append(f'      <tr style="background-color: {bg_color}; color: #f8fafc;">')
            padded = r + ["" for _ in range(len(clean_headers) - len(r))]
            for cell in padded[:len(clean_headers)]:
                html_lines.append(f'        <td style="border: 1px solid #334155; padding: 6px 14px; text-align: center; font-weight: 500;">{cell}</td>')
            html_lines.append('      </tr>')

        html_lines.append('    </tbody>')
        html_lines.append('  </table>')
        html_lines.append('</div>')
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
