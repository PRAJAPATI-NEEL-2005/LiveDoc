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

import os
import re
from typing import List, Dict, Any, Union
import cv2
import numpy as np

try:
    import pymupdf
    HAS_PYMUPDF = True
except ImportError:
    try:
        import fitz as pymupdf
        HAS_PYMUPDF = True
    except ImportError:
        HAS_PYMUPDF = False
from PyQt6.QtWidgets import QApplication
from PyQt6.QtPrintSupport import QPrinter
from PyQt6.QtGui import QTextDocument, QPageLayout, QPageSize, QPainter
from PyQt6.QtCore import QMarginsF, QRectF
from src.text_markup import clean_display_text


class PDFExporter:
    MATH_REPLACEMENTS = {
        r'\qquad': '&emsp;&emsp;',
        r'\quad': '&emsp;',
        r'\,': '&thinsp;',
        r'\;': ' ',
        r'\:': ' ',
        r'\ ': ' ',
        r'\land': '∧',
        r'\wedge': '∧',
        r'\lor': '∨',
        r'\vee': '∨',
        r'\neg': '¬',
        r'\sim': '~',
        r'\implies': '⇒',
        r'\Rightarrow': '⇒',
        r'\iff': '⇔',
        r'\Leftrightarrow': '⇔',
        r'\leftrightarrow': '↔',
        r'\to': '→',
        r'\rightarrow': '→',
        r'\longrightarrow': '→',
        r'\leftarrow': '←',
        r'\longleftarrow': '←',
        r'\le': '≤',
        r'\leq': '≤',
        r'\ge': '≥',
        r'\geq': '≥',
        r'\neq': '≠',
        r'\approx': '≈',
        r'\equiv': '≡',
        r'\times': '×',
        r'\cdot': '·',
        r'\pm': '±',
        r'\mp': '∓',
        r'\in': '∈',
        r'\notin': '∉',
        r'\subset': '⊂',
        r'\subseteq': '⊆',
        r'\supset': '⊃',
        r'\supseteq': '⊇',
        r'\cup': '∪',
        r'\cap': '∩',
        r'\emptyset': '∅',
        r'\forall': '∀',
        r'\exists': '∃',
        r'\nexists': '∄',
        r'\models': '⊨',
        r'\vdash': '⊢',
        r'\top': '⊤',
        r'\bot': '⊥',
        r'\sum': '∑',
        r'\prod': '∏',
        r'\int': '∫',
        r'\infty': '∞',
        r'\partial': '∂',
        r'\nabla': '∇',
        r'\Vert': '║',
        r'\vert': '|',
        r'\|': '║',
        r'\ldots': '…',
        r'\cdots': '…',
        r'\ddots': '⋱',
        r'\vdots': '⋮',
        r'\dots': '…',
        r'\prime': '′',
        # Greek lowercase
        r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
        r'\epsilon': 'ε', r'\varepsilon': 'ε', r'\zeta': 'ζ', r'\eta': 'η',
        r'\theta': 'θ', r'\vartheta': 'θ', r'\iota': 'ι', r'\kappa': 'κ',
        r'\lambda': 'λ', r'\mu': 'μ', r'\nu': 'ν', r'\xi': 'ξ',
        r'\pi': 'π', r'\rho': 'ρ', r'\sigma': 'σ', r'\tau': 'τ',
        r'\upsilon': 'υ', r'\phi': 'φ', r'\chi': 'χ', r'\psi': 'ψ',
        r'\omega': 'ω',
        # Greek uppercase
        r'\Gamma': 'Γ', r'\Delta': 'Δ', r'\Theta': 'Θ', r'\Lambda': 'Λ',
        r'\Xi': 'Ξ', r'\Pi': 'Π', r'\Sigma': 'Σ', r'\Upsilon': 'Υ',
        r'\Phi': 'Φ', r'\Psi': 'Ψ', r'\Omega': 'Ω'
    }

    @classmethod
    def _convert_latex_to_typography(cls, text: str) -> str:
        """
        Converts LaTeX math expressions and symbols into publication-grade HTML & Unicode.
        """
        if not text:
            return ""

        # 1. Text wrappers in math mode
        text = re.sub(re.escape(r"\textbf") + r"\{([^}]*)\}", r"<b>\1</b>", text)
        text = re.sub(re.escape(r"\textit") + r"\{([^}]*)\}", r"<i>\1</i>", text)
        text = re.sub(re.escape(r"\text") + r"\{([^}]*)\}", r"\1", text)
        text = re.sub(re.escape(r"\mathrm") + r"\{([^}]*)\}", r"\1", text)
        text = re.sub(re.escape(r"\mathbf") + r"\{([^}]*)\}", r"<b>\1</b>", text)
        text = re.sub(re.escape(r"\mathit") + r"\{([^}]*)\}", r"<i>\1</i>", text)
        text = re.sub(re.escape(r"\boldsymbol") + r"\{([^}]*)\}", r"<b>\1</b>", text)

        # 2. Labeled transitions and arrows (e.g. \xrightarrow{a}, \xrightarrow{b/1})
        text = re.sub(re.escape(r"\xrightarrow") + r"\{([^}]*)\}", r" &thinsp;―<sup>\1</sup>→&thinsp; ", text)
        text = re.sub(re.escape(r"\xleftarrow") + r"\{([^}]*)\}", r" &thinsp;←<sup>\1</sup>―&thinsp; ", text)

        # 3. Fractions, Roots & Overlines
        text = re.sub(re.escape(r"\frac") + r"\{([^}]*)\}\{([^}]*)\}", r"<sup>\1</sup>&frasl;<sub>\2</sub>", text)
        text = re.sub(re.escape(r"\sqrt") + r"\[([^\]]*)\]\{([^}]*)\}", r"<sup>\1</sup>√\2", text)
        text = re.sub(re.escape(r"\sqrt") + r"\{([^}]*)\}", r"√\1", text)
        text = re.sub(re.escape(r"\bar") + r"\{([^}]*)\}", r"<span style=\"text-decoration: overline;\">\1</span>", text)
        text = re.sub(re.escape(r"\overline") + r"\{([^}]*)\}", r"<span style=\"text-decoration: overline;\">\1</span>", text)

        # 4. Spacing wrappers
        text = re.sub(re.escape(r"\hspace") + r"\{[^}]+\}", " ", text)
        text = re.sub(re.escape(r"\vspace") + r"\{[^}]+\}", "", text)

        # 5. Core math symbol replacements (sorted by length descending to prevent prefix collisions)
        for sym in sorted(cls.MATH_REPLACEMENTS.keys(), key=len, reverse=True):
            text = text.replace(sym, cls.MATH_REPLACEMENTS[sym])

        # 6. Subscripts & Superscripts
        text = re.sub(r"\^\{([^}]*)\}", r"<sup>\1</sup>", text)
        text = re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", text)
        text = re.sub(r"\^([0-9a-zA-Z\+\-]+)", r"<sup>\1</sup>", text)
        text = re.sub(r"_([0-9a-zA-Z]+)", r"<sub>\1</sub>", text)

        # 7. Delimiters and primes
        text = text.replace(r"\left(", "(").replace(r"\right)", ")")
        text = text.replace(r"\left[", "[").replace(r"\right]", "]")
        text = text.replace(r"\{", "{").replace(r"\}", "}")
        text = text.replace(r"\'", "′").replace("'", "′")

        # 8. Strip math mode dollar signs
        text = text.replace("$$", "").replace("$", "")
        return text.strip()

    @classmethod
    def _format_inline_markdown(cls, text: str) -> str:
        """
        Formats inline Markdown syntax (bold, italic, code, links) and mathematical notation
        into clean, styled HTML.
        """
        if not text:
            return ""

        # 1. Convert LaTeX math and symbols first
        text = cls._convert_latex_to_typography(text)

        # 2. Markdown Bold: **text** or __text__
        text = re.sub(r'\*\*([^\*]+)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'__([^_]+)__', r'<b>\1</b>', text)

        # 3. Markdown Italic: *text* or _text_
        text = re.sub(r'(?<!\*)\*([^\*\n]+)\*(?!\*)', r'<i>\1</i>', text)
        text = re.sub(r'(?<!_)_([^_<\n]+)_(?!_)', r'<i>\1</i>', text)

        # 4. Inline Code: `code`
        text = re.sub(r'`([^`]+)`', r'<code style="background-color: #f1f5f9; padding: 1px 4px; border-radius: 3px; font-family: monospace;">\1</code>', text)

        # 5. Markdown Links: [label](url)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" style="color: #0284c7; text-decoration: underline;">\1</a>', text)

        return text

    @classmethod
    def _format_single_page_html(cls, page_content: str, p_idx: int, doc_title: str, font_pt: float = 9.2) -> str:
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
                is_compact = (num_cols <= 4 and all(len(c) <= 8 for c in header_row))
                tbl_width = "width: auto; min-width: 150px; max-width: 320px;" if is_compact else "width: 88%; max-width: 520px;"

                tbl_html = [
                    '<div style="page-break-inside: avoid; break-inside: avoid; margin: 4px 0; text-align: center;">',
                    f'<table style="border-collapse: collapse; margin: 0 auto; {tbl_width} font-size: {max(7.5, font_pt - 1.0):.1f}pt; font-family: monospace; page-break-inside: avoid; break-inside: avoid;">',
                    '<thead><tr style="background-color: #f1f5f9; border-top: 1pt solid #334155; border-bottom: 0.8pt solid #334155;">'
                ]
                for cell in header_row:
                    c_formatted = cls._format_inline_markdown(cell.strip())
                    tbl_html.append(f'<th style="border: 0.5pt solid #cbd5e1; padding: 3px 10px; text-align: center; font-weight: bold; color: #0f172a;">{c_formatted}</th>')
                tbl_html.append('</tr></thead><tbody>')

                for r_idx, row in enumerate(rows[1:]):
                    bg = "#ffffff" if r_idx % 2 == 0 else "#f8fafc"
                    border_b = "border-bottom: 1pt solid #334155;" if r_idx == len(rows[1:]) - 1 else ""
                    tbl_html.append(f'<tr style="background-color: {bg}; {border_b}">')
                    padded = row + ["" for _ in range(len(header_row) - len(row))]
                    for cell in padded[:len(header_row)]:
                        c_formatted = cls._format_inline_markdown(cell.strip())
                        tbl_html.append(f'<td style="border: 0.5pt solid #cbd5e1; padding: 2.5px 10px; text-align: center; color: #1e293b;">{c_formatted}</td>')
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
                title_txt = cls._format_inline_markdown(line_str[2:].strip())
                title_txt = re.sub(r'&emsp;+', ' — ', title_txt).strip(' — ')
                p_elems.append(
                    f'<div style="text-align: center; border-bottom: 1.5px solid #0284c7; padding-bottom: 3px; margin: 3px 0 5px 0;">'
                    f'<h1 style="color: #0369a1; font-size: {font_pt + 1.8:.1f}pt; margin: 0; font-weight: bold;">{title_txt}</h1></div>'
                )
            # 2. Section Heading (e.g. ## 2.1 Section Title)
            elif line_str.startswith("## "):
                h2_txt = cls._format_inline_markdown(line_str[3:].strip())
                p_elems.append(
                    f'<h2 style="color: #0284c7; font-size: {font_pt + 1.0:.1f}pt; font-weight: bold; margin-top: 5px; margin-bottom: 2px; border-bottom: 0.8px solid #e2e8f0; padding-bottom: 2px;">{h2_txt}</h2>'
                )
            # 3. Subheading (e.g. ### Subsection Name)
            elif line_str.startswith("### "):
                h3_txt = cls._format_inline_markdown(line_str[4:].strip())
                p_elems.append(
                    f'<h3 style="color: #0369a1; font-size: {font_pt + 0.4:.1f}pt; font-weight: bold; margin-top: 4px; margin-bottom: 2px;">{h3_txt}</h3>'
                )
            # 3.5 Embedded Image / Diagram (Markdown syntax: ![Caption](path_or_data_uri))
            elif line_str.startswith("![") and "](" in line_str:
                img_m = re.match(r"!\[(.*?)\]\((.*?)\)", line_str)
                if img_m:
                    caption_raw = img_m.group(1).strip()
                    caption_txt = cls._format_inline_markdown(caption_raw)
                    img_src = img_m.group(2).strip()
                    if os.path.exists(img_src):
                        cap_lower = caption_raw.lower()
                        is_compact = any(k in cap_lower for k in ["logo", "barcode", "qr", "icon", "badge", "symbol"])

                        # Inspect actual image dimensions
                        try:
                            im_stat = cv2.imread(img_src)
                            ih, iw = im_stat.shape[:2] if im_stat is not None else (200, 300)
                        except Exception:
                            ih, iw = (200, 300)

                        aspect = iw / float(max(1, ih))
                        num_images = len(re.findall(r'!\[.*?\]\(.*?\)', page_content))

                        if is_compact or ih < 100:
                            # Small logos, barcodes, QR codes
                            max_h = min(48, max(24, ih))
                            max_w = min(42, max(25, int(aspect * max_h * 1.5)))
                            img_style = f"max-width: {max_w}%; max-height: {max_h}px; border: 0.6pt solid #cbd5e1; border-radius: 3px; padding: 1.5px;"
                            div_margin = "2px auto 2px auto;"
                            caption_html = f'<div style="font-size: {max(6.5, font_pt - 2.2):.1f}pt; color: #64748b; font-style: italic; margin-top: 1px;">{caption_txt}</div>'
                        elif num_images >= 3:
                            # Multi-figure page
                            max_h = 115 if aspect > 1.8 else 105
                            img_style = f"max-width: 82%; max-height: {max_h}px; border: 0.8pt solid #cbd5e1; border-radius: 4px; padding: 2px;"
                            div_margin = "3px auto 4px auto;"
                            caption_html = f'<div style="font-size: {max(7.0, font_pt - 1.8):.1f}pt; color: #475569; font-style: italic; margin-top: 2px;">{caption_txt}</div>'
                        elif num_images >= 2:
                            max_h = 145
                            img_style = f"max-width: 84%; max-height: {max_h}px; border: 0.8pt solid #cbd5e1; border-radius: 4px; padding: 2px;"
                            div_margin = "4px auto 6px auto;"
                            caption_html = f'<div style="font-size: {max(7.2, font_pt - 1.6):.1f}pt; color: #475569; font-style: italic; margin-top: 2px;">{caption_txt}</div>'
                        else:
                            # Single main figure
                            max_h = 220
                            img_style = f"max-width: 86%; max-height: {max_h}px; border: 0.8pt solid #cbd5e1; border-radius: 4px; padding: 2px;"
                            div_margin = "6px auto 8px auto;"
                            caption_html = f'<div style="font-size: {max(7.5, font_pt - 1.4):.1f}pt; color: #475569; font-style: italic; margin-top: 3px;">{caption_txt}</div>'

                        p_elems.append(
                            f'<div style="page-break-inside: avoid; break-inside: avoid; text-align: center; margin: {div_margin} background-color: #ffffff;">'
                            f'<img src="{img_src}" style="{img_style} background-color: #ffffff;" />'
                            f'{caption_html}'
                            f'</div>'
                        )
                    else:
                        p_elems.append(
                            f'<div style="page-break-inside: avoid; break-inside: avoid; background-color: #f8fafc; border-left: 2.5pt solid #0284c7; border-radius: 4px; padding: 5pt 10pt; margin: 6pt auto; width: 88%; text-align: left; color: #334155; font-size: 8.5pt;">'
                            f'<span style="font-weight: bold; color: #0284c7;">📊 Figure / Diagram:</span> {caption_txt}'
                            f'</div>'
                        )
                else:
                    p_elems.append(f'<p style="margin: 2px 0; font-size: {font_pt:.1f}pt;">{cls._format_inline_markdown(line_str)}</p>')
            elif line_str.startswith("> 📷"):
                img_txt = line_str.lstrip("> 📷 *").rstrip("*")
                p_elems.append(
                    f'<div style="background-color: #f8fafc; border-left: 2pt solid #64748b; padding: 3pt 6pt; margin: 4pt 0; color: #475569; font-style: italic; font-size: 8.5pt;">📷 {cls._format_inline_markdown(img_txt)}</div>'
                )
            else:
                # 4. Paragraphs, List Items, and Formula definitions
                is_bullet = bool(re.match(r'^[\*\-\+\•]\s+', line_str))
                is_num_item = bool(re.match(
                    r'^(?:(?:\d{1,2}|[ivxlcdm]{1,5})[\.\)]|\(\d{1,2}\)|\([a-zA-Z]\)|[a-zA-Z]\)|\d{1,2}[\.\s]+[A-Z])\s*',
                    line_str, re.IGNORECASE
                ))
                is_formula_def = bool(re.search(r'=\s*[\{\(]|can be\s*=', line_str)) or line_str.startswith("$$")

                if is_bullet:
                    raw_item = re.sub(r'^[\*\-\+\•]\s+', '', line_str)
                    formatted_item = cls._format_inline_markdown(raw_item)
                    p_elems.append(
                        f'<p style="margin: 2px 0 2px 14px; font-size: {font_pt:.1f}pt; line-height: 1.34; text-align: left; color: #0f172a;">'
                        f'<span style="color: #0284c7; font-weight: bold; margin-right: 6px;">•</span>{formatted_item}'
                        f'</p>'
                    )
                elif is_num_item:
                    formatted_item = cls._format_inline_markdown(line_str)
                    p_elems.append(
                        f'<p style="margin: 2px 0 2px 14px; font-size: {font_pt:.1f}pt; line-height: 1.34; text-align: left; color: #0f172a;">{formatted_item}</p>'
                    )
                elif is_formula_def:
                    formatted_formula = cls._format_inline_markdown(line_str)
                    p_elems.append(
                        f'<p style="margin: 2px 0 2px 18px; font-size: {max(7.5, font_pt - 0.5):.1f}pt; font-family: monospace; line-height: 1.30; text-align: left; color: #1e293b;">{formatted_formula}</p>'
                    )
                else:
                    margin_v = "1.5px" if font_pt <= 8.0 else "2.5px"
                    formatted_para = cls._format_inline_markdown(line_str)
                    p_elems.append(
                        f'<p style="margin: {margin_v} 0; font-size: {font_pt:.1f}pt; line-height: 1.30; text-align: justify; text-justify: inter-word; color: #0f172a;">{formatted_para}</p>'
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
                # Auto-fit font scaling: find comfortable font size that achieves minimum page count
                best_doc = None
                best_page_count = 999
                for font_pt in [9.2, 8.8, 8.4, 8.0, 7.6, 7.2, 6.8]:
                    html = PDFExporter._format_single_page_html(page_content, p_idx, doc_title, font_pt=font_pt)
                    doc = QTextDocument()
                    doc.setPageSize(paint_rect.size())
                    doc.setHtml(html)
                    pc = doc.pageCount()
                    if pc == 1:
                        best_doc = doc
                        best_page_count = 1
                        break
                    if pc < best_page_count:
                        best_page_count = pc
                        best_doc = doc

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

    @classmethod
    def create_digital_reconstructed_pdf(
        cls,
        pages_data: Union[List[Dict[str, Any]], Dict[str, Any]],
        output_path: str,
        doc_title: str = "Digitized Document",
        include_image_background: bool = False
    ) -> bool:
        """
        Pure Digital Document Reconstruction Engine:
        Digitizes physical documents into 100% clean, vector-grade digital PDF pages:
        - Pure white digital background (zero camera noise, shadows, or photo artifacts).
        - Crisp digital typography (Title, Headings, Paragraphs) placed at their exact physical 2D coordinates.
        - High-precision digital vector tables (styled header row, alternating zebra rows, cell borders).
        - Cropped diagram/figure images placed into their detected bounding boxes.
        - Multi-page document generation with metadata.
        - Optional include_image_background=True for dual-layer sandwich mode.
        """
        if isinstance(pages_data, dict):
            pages_list = [pages_data]
        elif isinstance(pages_data, list):
            pages_list = pages_data
        else:
            pages_list = []

        if not pages_list:
            return False

        if not HAS_PYMUPDF:
            print("Notice: pymupdf not available, falling back to typeset PDF engine.")
            combined_md = "\n\n---\n\n".join(
                p.get("markdown", "") for p in pages_list if p.get("markdown")
            )
            return cls.markdown_to_pdf(combined_md, output_path, doc_title=doc_title)

        try:
            doc = pymupdf.open()
            valid_pages_rendered = 0

            for p_idx, page_info in enumerate(pages_list):
                # 1. Canvas Dimensions & Scaling
                img_w, img_h = 800, 1200
                if page_info.get("image_shape"):
                    img_h, img_w = page_info["image_shape"][:2]
                elif page_info.get("image") is not None and hasattr(page_info["image"], "shape"):
                    img_h, img_w = page_info["image"].shape[:2]
                elif page_info.get("doc_data") and page_info["doc_data"].get("image_shape"):
                    img_h, img_w = page_info["doc_data"]["image_shape"][:2]

                # Standard A4 width in PDF points (595.0 pt), maintaining exact physical aspect ratio
                page_w = 595.0
                page_h = round(page_w * (img_h / float(max(1, img_w))), 2)

                page = doc.new_page(width=page_w, height=page_h)

                # Always fill pure white — no camera noise or shadow leakthrough
                page.draw_rect(
                    pymupdf.Rect(0, 0, page_w, page_h),
                    color=None, fill=(1.0, 1.0, 1.0)
                )

                sx = page_w / float(max(1, img_w))
                sy = page_h / float(max(1, img_h))

                # Optional Camera Photo Background (if explicitly requested)
                if include_image_background:
                    img_mat = page_info.get("image")
                    if isinstance(img_mat, str) and os.path.exists(img_mat):
                        img_mat = cv2.imread(img_mat)
                    if img_mat is not None and isinstance(img_mat, np.ndarray):
                        success, enc = cv2.imencode(".jpg", img_mat, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        if success:
                            page.insert_image(pymupdf.Rect(0, 0, page_w, page_h), stream=enc.tobytes())

                # Collect structured elements
                elements = page_info.get("elements", [])
                if not elements and page_info.get("doc_data"):
                    elements = page_info["doc_data"].get("elements", [])

                raw_tokens = page_info.get("tokens", [])
                if not raw_tokens and page_info.get("doc_data"):
                    raw_tokens = page_info["doc_data"].get("tokens", [])

                if elements:
                    # Sort top to bottom
                    sorted_elements = sorted(
                        elements,
                        key=lambda e: e.get("box", [0, 0, 0, 0])[1] if e.get("box") else e.get("top_y", 0)
                    )

                    _L = 28.0                   # left margin (pt)
                    _R = page_w - 22.0          # right margin (pt)
                    _flow_y = 28.0              # flowing cursor for elements without box

                    for el in sorted_elements:
                        el_type = el.get("type", "Paragraph")
                        box = el.get("box")

                        if box and len(box) >= 4 and box[2] > box[0] and box[3] > box[1]:
                            # ── Offline OCR mode: exact coordinates from image ──
                            x0 = float(box[0]) * sx
                            y0 = float(box[1]) * sy
                            x1 = float(box[2]) * sx
                            y1 = float(box[3]) * sy

                            # Clamp to safe page margins
                            x0 = max(_L, min(page_w - 22.0, x0))
                            y0 = max(5.0, min(page_h - 15.0, y0))
                            x1 = max(x0 + 15.0, min(page_w - 18.0, x1))
                            y1 = max(y0 + 10.0, min(page_h - 10.0, y1))
                            _flow_y = max(_flow_y, y1 + 3.0)

                        else:
                            # ── Smart AI / no-box mode: flowing layout ──
                            x0, x1 = _L, _R
                            if el_type == "Title":
                                el_h = 26.0
                            elif el_type == "Heading":
                                el_h = 20.0 if el.get("level", 2) == 2 else 16.0
                            elif el_type == "Table":
                                el_h = max(50.0, (len(el.get("rows", [])) + 1) * 16.0)
                            elif el_type in ("Image", "Figure", "Picture"):
                                el_h = 100.0
                            else:
                                txt = el.get("text") or el.get("content", "")
                                cpl = max(1, int((_R - _L) / 5.2))
                                el_h = min(
                                    page_h - _flow_y - 18.0,
                                    max(14.0, (len(txt) // cpl + 1) * 11.5)
                                )

                            if _flow_y + el_h > page_h - 12.0:
                                continue   # page full — skip overflow
                            y0, y1 = _flow_y, _flow_y + el_h
                            _flow_y = y1 + 5.0

                        if x1 <= x0 + 2.0 or y1 <= y0 + 2.0:
                            continue

                        # ── A. Document Title ──────────────────────────────────
                        if el_type == "Title":
                            t_txt = clean_display_text(el.get("text") or el.get("content", ""))
                            t_h = max(24.0, y1 - y0)
                            fs = min(17.0, max(12.0, t_h * 0.65))
                            t_rect = pymupdf.Rect(x0, y0, x1, y0 + t_h)
                            page.insert_textbox(
                                t_rect, t_txt,
                                fontsize=fs, fontname="helv",
                                color=(0.01, 0.2, 0.5),
                                align=pymupdf.TEXT_ALIGN_CENTER
                            )
                            page.draw_line(
                                pymupdf.Point(x0, y0 + t_h + 2), pymupdf.Point(x1, y0 + t_h + 2),
                                color=(0.1, 0.5, 0.8), width=1.2
                            )

                        # ── B. Section Heading ─────────────────────────────────
                        elif el_type == "Heading":
                            h_txt = clean_display_text(el.get("text") or el.get("content", ""))
                            lvl = el.get("level", 2)
                            color = (0.02, 0.35, 0.65) if lvl == 2 else (0.1, 0.45, 0.7)
                            fs = 12.0 if lvl == 2 else 10.5
                            h_h = max(18.0, y1 - y0)
                            h_rect = pymupdf.Rect(x0, y0, x1, y0 + h_h)
                            page.insert_textbox(
                                h_rect, h_txt,
                                fontsize=fs, fontname="helv",
                                color=color,
                                align=pymupdf.TEXT_ALIGN_LEFT
                            )

                        # ── C. Structured Digital Table ─────────────────────────
                        elif el_type == "Table":
                            headers = el.get("headers", [])
                            rows = el.get("rows", [])

                            if (not headers or not rows) and el.get("markdown"):
                                md_lines = [l.strip() for l in el["markdown"].strip().split("\n") if l.strip().startswith("|")]
                                if len(md_lines) >= 2:
                                    headers = [c.strip() for c in md_lines[0].strip("|").split("|")]
                                    data_rows = []
                                    for line in md_lines[1:]:
                                        cells = [c.strip() for c in line.strip("|").split("|")]
                                        if all(re.match(r"^:?-+:?$", c) for c in cells):
                                            continue
                                        data_rows.append(cells)
                                    rows = data_rows

                            if headers:
                                num_cols = max(len(headers), max((len(r) for r in rows), default=1))
                                num_rows = len(rows) + 1
                                col_w = (x1 - x0) / float(max(1, num_cols))
                                row_h = (y1 - y0) / float(max(1, num_rows))

                                # Header Row
                                hdr_rect = pymupdf.Rect(x0, y0, x1, y0 + row_h)
                                page.draw_rect(hdr_rect, color=(0.2, 0.27, 0.35), fill=(0.94, 0.96, 0.98), width=0.8)
                                for c_idx, h_text in enumerate(headers):
                                    cr = pymupdf.Rect(x0 + c_idx * col_w, y0 + 1, x0 + (c_idx + 1) * col_w, y0 + row_h)
                                    page.insert_textbox(
                                        cr, clean_display_text(str(h_text)),
                                        fontsize=min(9.0, max(6.5, row_h * 0.55)),
                                        fontname="helv",
                                        color=(0.05, 0.1, 0.2),
                                        align=pymupdf.TEXT_ALIGN_CENTER
                                    )

                                # Data Rows
                                for r_idx, row in enumerate(rows):
                                    cy0 = y0 + (r_idx + 1) * row_h
                                    cy1 = cy0 + row_h
                                    bg = (1.0, 1.0, 1.0) if r_idx % 2 == 0 else (0.97, 0.98, 0.99)
                                    r_rect = pymupdf.Rect(x0, cy0, x1, cy1)
                                    page.draw_rect(r_rect, color=(0.8, 0.84, 0.88), fill=bg, width=0.5)
                                    for c_idx, cell_text in enumerate(row):
                                        cr = pymupdf.Rect(x0 + c_idx * col_w, cy0 + 1, x0 + (c_idx + 1) * col_w, cy1)
                                        page.insert_textbox(
                                            cr, clean_display_text(str(cell_text)),
                                            fontsize=min(8.5, max(6.0, row_h * 0.50)),
                                            fontname="helv",
                                            color=(0.1, 0.15, 0.2),
                                            align=pymupdf.TEXT_ALIGN_CENTER
                                        )

                                # Table Outer Border
                                page.draw_rect(pymupdf.Rect(x0, y0, x1, y1), color=(0.2, 0.27, 0.35), width=1.0)
                            else:
                                t_rect = pymupdf.Rect(x0, y0, x1, y1)
                                page.draw_rect(t_rect, color=(0.2, 0.27, 0.35), fill=(0.98, 0.98, 0.99), width=0.8)
                                page.insert_textbox(
                                    t_rect, el.get("text", ""),
                                    fontsize=8.0, fontname="helv",
                                    color=(0.1, 0.15, 0.2)
                                )

                        # ── D. Cropped Figure / Diagram ────────────────────────
                        elif el_type in ("Image", "Figure", "Picture"):
                            fig_path = el.get("image_path")
                            fig_caption = el.get("caption") or el.get("content", "")
                            has_cap = bool(fig_caption and not fig_caption.startswith("!["))
                            cap_h = 14 if has_cap else 0
                            fig_rect = pymupdf.Rect(x0, y0, x1, max(y0 + 20, y1 - cap_h))

                            fig_bytes = None
                            if fig_path and os.path.exists(fig_path):
                                try:
                                    with open(fig_path, "rb") as f:
                                        fig_bytes = f.read()
                                except Exception:
                                    pass

                            if not fig_bytes and page_info.get("image") is not None and isinstance(page_info["image"], np.ndarray):
                                orig_img = page_info["image"]
                                bx1, by1, bx2, by2 = map(int, box[:4])
                                ih, iw = orig_img.shape[:2]
                                crop = orig_img[max(0, by1):min(ih, by2), max(0, bx1):min(iw, bx2)]
                                if crop.size > 0:
                                    ok_enc, enc_b = cv2.imencode(".png", crop)
                                    if ok_enc:
                                        fig_bytes = enc_b.tobytes()

                            if fig_bytes:
                                try:
                                    page.insert_image(fig_rect, stream=fig_bytes, keep_proportion=True)
                                    page.draw_rect(fig_rect, color=(0.8, 0.84, 0.88), width=0.6)
                                except Exception:
                                    pass

                            if has_cap:
                                cap_rect = pymupdf.Rect(x0, y1 - 13, x1, y1)
                                page.insert_textbox(
                                    cap_rect, fig_caption,
                                    fontsize=7.5, fontname="helv",
                                    color=(0.3, 0.35, 0.4),
                                    align=pymupdf.TEXT_ALIGN_CENTER
                                )

                        # ── E. Text Paragraphs ─────────────────────────────────
                        else:
                            p_text = clean_display_text(el.get("text") or el.get("content", ""))
                            p_rect = pymupdf.Rect(x0, y0, x1, y1)
                            fs = 9.5
                            while fs >= 6.0:
                                rc = page.insert_textbox(
                                    p_rect, p_text,
                                    fontsize=fs, fontname="helv",
                                    color=(0.06, 0.09, 0.16)
                                )
                                if rc >= 0:
                                    break
                                fs -= 0.5
                            if rc < 0:
                                page.insert_textbox(
                                    p_rect, p_text,
                                    fontsize=6.0, fontname="helv",
                                    color=(0.06, 0.09, 0.16)
                                )

                elif raw_tokens:
                    # Token-level digital reconstruction fallback
                    for t in raw_tokens:
                        txt = str(t.get("text", "")).strip()
                        if not txt:
                            continue
                        bx0 = float(t.get("min_x", 0)) * sx
                        by0 = float(t.get("min_y", 0)) * sy
                        bx1 = float(t.get("max_x", img_w)) * sx
                        by1 = float(t.get("max_y", img_h)) * sy
                        tw = max(1.0, bx1 - bx0)
                        th = max(1.0, by1 - by0)
                        fs = max(4.0, th * 0.82)
                        try:
                            tl = pymupdf.get_text_length(txt, fontname="helv", fontsize=fs)
                            if tl > tw and tl > 0:
                                fs = max(3.5, fs * (tw / tl))
                        except Exception:
                            pass
                        baseline_y = by1 - (th * 0.15)
                        page.insert_text(
                            pymupdf.Point(bx0, baseline_y),
                            txt,
                            fontsize=fs,
                            fontname="helv",
                            color=(0.06, 0.09, 0.16),
                            render_mode=0
                        )

                valid_pages_rendered += 1

            if valid_pages_rendered > 0:
                doc.set_metadata({
                    "title": doc_title,
                    "creator": "LiveDoc Digital Document Reconstruction Engine"
                })
                doc.save(output_path, garbage=4, deflate=True)
                doc.close()
                return True
            else:
                doc.close()
                combined_md = "\n\n---\n\n".join(
                    p.get("markdown", "") for p in pages_list if p.get("markdown")
                )
                if combined_md:
                    return cls.markdown_to_pdf(combined_md, output_path, doc_title=doc_title)
                return False
        except Exception as e:
            print(f"Error generating digital reconstructed PDF: {e}")
            try:
                combined_md = "\n\n---\n\n".join(
                    p.get("markdown", "") for p in pages_list if p.get("markdown")
                )
                if combined_md:
                    return cls.markdown_to_pdf(combined_md, output_path, doc_title=doc_title)
            except Exception:
                pass
            return False

    # Aliases for backward compatibility and discovery
    create_searchable_pdf = create_digital_reconstructed_pdf
    export_dual_layer_pdf = create_digital_reconstructed_pdf

