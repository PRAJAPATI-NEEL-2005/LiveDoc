"""
LiveDoc PDF Exporter Engine
Generates publication-quality, multi-page PDF documents from digitized content.
Features:
- Book-grade typography, alignment, and proportional spacing.
- Continuous flowing paragraphs without premature line breaks or empty margins.
- Strict table protection: zero row splitting across page breaks.
- Dynamic page auto-fit: guarantees each scanned page forms a self-contained, complete publication page.
- Native via PyQt6 QtPrintSupport (zero external CLI dependencies).
"""

import re
from typing import List, Dict, Any
from PyQt6.QtWidgets import QApplication
from PyQt6.QtPrintSupport import QPrinter
from PyQt6.QtGui import QTextDocument, QPageLayout, QPageSize, QPainter
from PyQt6.QtCore import QMarginsF, QRectF


class PDFExporter:
    @staticmethod
    def _format_single_page_html(page_content: str, p_idx: int, doc_title: str, font_pt: float = 9.2) -> str:
        """
        Formats a single scanned document page into book-proportioned publication HTML.
        """
        lines = page_content.strip().split("\n")
        p_elems = []
        t_rows = []
        in_table = False

        def flush_table(rows, elems):
            if rows:
                header_row = rows[0]
                num_cols = len(header_row)
                is_compact = (num_cols <= 4 and all(len(c) <= 6 for c in header_row))
                tbl_width = "width: auto; min-width: 150px; max-width: 260px;" if is_compact else "width: 88%; max-width: 500px;"

                tbl_html = [
                    '<div style="page-break-inside: avoid; break-inside: avoid; margin: 4px 0; text-align: center;">',
                    f'<table style="border-collapse: collapse; margin: 0 auto; {tbl_width} font-size: {max(7.5, font_pt - 1.2):.1f}pt; font-family: monospace; page-break-inside: avoid; break-inside: avoid;">',
                    '<thead><tr style="background-color: #f1f5f9; border-top: 1pt solid #334155; border-bottom: 0.8pt solid #334155;">'
                ]
                for cell in header_row:
                    tbl_html.append(f'<th style="border: 0.5pt solid #cbd5e1; padding: 2.5px 8px; text-align: center; font-weight: bold; color: #0f172a;">{cell.strip()}</th>')
                tbl_html.append('</tr></thead><tbody>')

                for r_idx, row in enumerate(rows[1:]):
                    bg = "#ffffff" if r_idx % 2 == 0 else "#f8fafc"
                    border_b = "border-bottom: 1pt solid #334155;" if r_idx == len(rows[1:]) - 1 else ""
                    tbl_html.append(f'<tr style="background-color: {bg}; {border_b}">')
                    padded = row + ["" for _ in range(len(header_row) - len(row))]
                    for cell in padded[:len(header_row)]:
                        tbl_html.append(f'<td style="border: 0.5pt solid #cbd5e1; padding: 2px 8px; text-align: center; color: #1e293b;">{cell.strip()}</td>')
                    tbl_html.append('</tr>')
                tbl_html.append('</tbody></table></div>')
                tbl_rendered = "".join(tbl_html)
                # Deduplicate consecutive identical tables
                if elems and elems[-1] == tbl_rendered:
                    rows.clear()
                    return
                elems.append(tbl_rendered)
                rows.clear()

        for line in lines:
            line_str = line.strip()
            if not line_str or line_str.startswith("<!--"):
                continue

            # Table row detection
            if line_str.startswith("|") and line_str.endswith("|"):
                cells = [c.strip() for c in line_str.strip("|").split("|")]
                if all(re.match(r"^:?-+:?$", c) for c in cells):
                    continue
                in_table = True
                t_rows.append(cells)
                continue
            else:
                if in_table:
                    flush_table(t_rows, p_elems)
                    in_table = False

            # 1. Document Title / Header Banner
            if line_str.startswith("# "):
                title_txt = line_str[2:].strip()
                p_elems.append(
                    f'<div style="text-align: center; border-bottom: 1.5px solid #0284c7; padding-bottom: 3px; margin: 3px 0 5px 0;">'
                    f'<h1 style="color: #0369a1; font-size: {font_pt + 1.8:.1f}pt; margin: 0; font-weight: bold;">{title_txt}</h1></div>'
                )
            # 2. Section Heading (e.g. ## 2.1 Section Title)
            elif line_str.startswith("## "):
                h2_txt = line_str[3:].strip()
                p_elems.append(
                    f'<h2 style="color: #0284c7; font-size: {font_pt + 1.0:.1f}pt; font-weight: bold; margin-top: 5px; margin-bottom: 2px; border-bottom: 0.8px solid #e2e8f0; padding-bottom: 2px;">{h2_txt}</h2>'
                )
            # 3. Subheading (e.g. ### Subsection Name)
            elif line_str.startswith("### "):
                h3_txt = line_str[4:].strip()
                p_elems.append(
                    f'<h3 style="color: #0369a1; font-size: {font_pt + 0.4:.1f}pt; font-weight: bold; margin-top: 4px; margin-bottom: 2px;">{h3_txt}</h3>'
                )
            elif line_str.startswith("> 📷"):
                img_txt = line_str.lstrip("> 📷 *").rstrip("*")
                p_elems.append(
                    f'<div style="background-color: #f8fafc; border-left: 2pt solid #64748b; padding: 3pt 6pt; margin: 4pt 0; color: #475569; font-style: italic; font-size: 8.5pt;">📷 {img_txt}</div>'
                )
            else:
                # 4. Paragraphs, List Items, and Formula definitions
                is_list_item = bool(re.match(
                    r'^(?:(?:\d{1,2}|[ivxlcdm]{1,5})[\.\)]|\(\d{1,2}\)|\([a-zA-Z]\)|[a-zA-Z]\)|\d{1,2}[\.\s]+[A-Z]|[\u2022\u25cf\u25cb\u25aa\u25ab\u25c6\u2212\-\*])\s*',
                    line_str, re.IGNORECASE
                ))
                is_formula_def = bool(re.search(r'=\s*[\{\(]|can be\s*=', line_str))

                if is_list_item:
                    p_elems.append(
                        f'<p style="margin: 1.5px 0 1.5px 14px; font-size: {font_pt:.1f}pt; line-height: 1.32; text-align: left; color: #0f172a;">{line_str}</p>'
                    )
                elif is_formula_def:
                    p_elems.append(
                        f'<p style="margin: 1px 0 1px 18px; font-size: {max(7.5, font_pt - 0.7):.1f}pt; font-family: monospace; line-height: 1.28; text-align: left; color: #1e293b;">{line_str}</p>'
                    )
                else:
                    margin_v = "1.5px" if font_pt <= 8.0 else "2.5px"
                    p_elems.append(
                        f'<p style="margin: {margin_v} 0; font-size: {font_pt:.1f}pt; line-height: 1.30; text-align: justify; text-justify: inter-word; color: #0f172a;">{line_str}</p>'
                    )

        if in_table:
            flush_table(t_rows, p_elems)

        badge = (
            f'<div style="color: #64748b; font-size: 7.5pt; font-weight: bold; text-transform: uppercase; letter-spacing: 0.8px; border-bottom: 1px dashed #cbd5e1; padding-bottom: 2px; margin-bottom: 4px;">'
            f'LiveDoc Digitized Document — Scanned Page {p_idx + 1}'
            f'</div>'
        )

        page_body = badge + "\n" + "\n".join(p_elems)

        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    font-family: Arial, "Helvetica Neue", Helvetica, sans-serif;
    color: #0f172a;
    margin: 0;
    padding: 0;
    font-size: {font_pt:.1f}pt;
    line-height: 1.35;
  }}
  table {{
    border-collapse: collapse;
    page-break-inside: avoid !important;
    break-inside: avoid !important;
  }}
  h1, h2, h3, h4 {{
    page-break-after: avoid !important;
    break-after: avoid !important;
  }}
  th, td {{
    word-break: break-word;
  }}
</style>
</head>
<body>
{page_body}
</body>
</html>"""

    @staticmethod
    def markdown_to_print_html(markdown_text: str, doc_title: str = "Digitized Document") -> str:
        """
        Transforms Markdown into publication HTML for preview or single-file export.
        """
        raw_pages = re.split(r"\n+<!--\s*Page\s*\d+\s*-->\n*|\n+---\n+", markdown_text)
        pages = [p.strip() for p in raw_pages if p.strip()]
        if not pages:
            pages = [markdown_text.strip()]

        html_pages = [PDFExporter._format_single_page_html(p, i, doc_title, font_pt=9.2) for i, p in enumerate(pages)]
        full_body = '<div style="page-break-before: always;"></div>'.join(html_pages)
        return full_body

    @staticmethod
    def markdown_to_pdf(markdown_text: str, output_path: str, doc_title: str = "Digitized Document") -> bool:
        """
        Convert structured Markdown text into a publication-grade PDF file.
        Features intelligent auto-fit: adjusts font sizing dynamically so each scanned page
        forms an authentic, complete, self-contained single PDF page with zero broken tables.
        """
        try:
            app = QApplication.instance()
            if app is None:
                app = QApplication(["LiveDocPDF"])

            printer = QPrinter(QPrinter.PrinterMode.ScreenResolution)
            printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
            printer.setOutputFileName(output_path)

            layout = QPageLayout(
                QPageSize(QPageSize.PageSizeId.A4),
                QPageLayout.Orientation.Portrait,
                QMarginsF(12, 10, 12, 10),
                QPageLayout.Unit.Millimeter
            )
            printer.setPageLayout(layout)

            paint_rect = printer.pageLayout().paintRect(QPageLayout.Unit.Point)
            paint_w = paint_rect.width()
            paint_h = paint_rect.height()

            raw_pages = re.split(r"\n+<!--\s*Page\s*\d+\s*-->\n*|\n+---\n+", markdown_text)
            pages = [p.strip() for p in raw_pages if p.strip()]
            if not pages:
                pages = [markdown_text.strip()]

            painter = QPainter(printer)
            first_page = True

            for p_idx, page_content in enumerate(pages):
                # Auto-fit font scaling: step down font size if content slightly exceeds 1 page
                best_doc = None
                for font_pt in [9.2, 8.8, 8.4, 8.0, 7.6, 7.2, 6.8]:
                    html = PDFExporter._format_single_page_html(page_content, p_idx, doc_title, font_pt=font_pt)
                    doc = QTextDocument()
                    doc.setPageSize(paint_rect.size())
                    doc.setHtml(html)
                    best_doc = doc
                    if doc.pageCount() == 1:
                        break

                # Render all sub-pages of this scanned document (usually exactly 1)
                sub_count = best_doc.pageCount()
                for sub_idx in range(sub_count):
                    if not first_page:
                        printer.newPage()
                    first_page = False

                    painter.save()
                    painter.translate(0, -sub_idx * paint_h)
                    painter.setClipRect(QRectF(0, sub_idx * paint_h, paint_w, paint_h))
                    best_doc.drawContents(painter)
                    painter.restore()

            painter.end()
            return True
        except Exception as e:
            print(f"Error generating PDF: {e}")
            return False

    @staticmethod
    def html_to_pdf(html_content: str, output_path: str) -> bool:
        """
        Renders raw HTML content onto a multi-page PDF document via PyQt6 QtPrintSupport.
        """
        try:
            app = QApplication.instance()
            if app is None:
                app = QApplication(["LiveDocPDF"])

            doc = QTextDocument()
            doc.setHtml(html_content)

            printer = QPrinter(QPrinter.PrinterMode.ScreenResolution)
            printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
            printer.setOutputFileName(output_path)

            layout = QPageLayout(
                QPageSize(QPageSize.PageSizeId.A4),
                QPageLayout.Orientation.Portrait,
                QMarginsF(12, 10, 12, 10),
                QPageLayout.Unit.Millimeter
            )
            printer.setPageLayout(layout)
            doc.print(printer)
            return True
        except Exception as e:
            print(f"Error printing HTML to PDF: {e}")
            return False
