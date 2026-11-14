"""
LiveDoc Layout Renderer
Reconstructs a scanned page as a clean digital page, placing every detected
text line / table / figure at the position it occupied on the physical page.

The same painter code drives both the on-screen "Visual 2D Layout" tab and the
exported PDF, so what the user sees on screen is what ends up in the file.

- White background only (the camera photo is never drawn).
- Text lines are drawn at their own bounding boxes (font size derived from the
  line height, shrunk if needed so the line never overflows its original width).
- Style per element type: Title / Heading / Paragraph / Table / Image.
- Elements without coordinates (e.g. Smart AI output) flow top-to-bottom.
"""

import os
import re
from typing import Any, Dict, List, Tuple

from PyQt6.QtCore import Qt, QRectF, QSizeF, QMarginsF
from PyQt6.QtGui import (
    QColor, QFont, QFontMetricsF, QImage, QPainter, QPen, QBrush,
    QPageSize, QPdfWriter, QPixmap
)

from src.text_markup import clean_display_text

FONT_FAMILY = "Arial"

COLOR_TITLE = QColor("#0369a1")
COLOR_H2 = QColor("#0284c7")
COLOR_H3 = QColor("#0369a1")
COLOR_BODY = QColor("#0f172a")
COLOR_TABLE_LINE = QColor("#334155")
COLOR_TABLE_HEAD_BG = QColor("#e8eef5")
COLOR_TABLE_ALT_BG = QColor("#f8fafc")

PDF_PAGE_WIDTH_PT = 595.0


def _font(px: float, bold: bool = False) -> QFont:
    f = QFont(FONT_FAMILY)
    f.setPixelSize(max(5, int(round(px))))
    f.setBold(bold)
    return f


def _fit_font(text: str, box_w: float, box_h: float, bold: bool,
              height_ratio: float = 0.78, min_scale: float = 0.45) -> Tuple[QFont, float]:
    """Font sized from the line height, shrunk so the text never exceeds box_w."""
    px = max(6.0, box_h * height_ratio)
    font = _font(px, bold)
    tw = QFontMetricsF(font).horizontalAdvance(text)
    if tw > box_w > 0:
        px = max(5.0, px * max(min_scale, box_w / tw))
        font = _font(px, bold)
        tw = QFontMetricsF(font).horizontalAdvance(text)
    return font, tw


def _draw_line(p: QPainter, text: str, box: List[float], bold: bool, color: QColor,
               align: str = "left", stretch: bool = False, fixed_px: float = None):
    text = clean_display_text(text)
    if not text:
        return
    x1, y1, x2, y2 = box
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    if fixed_px:
        # Consistent typography: same size for every line of a kind, shrunk only if it overflows its box
        font = _font(fixed_px, bold)
        tw = QFontMetricsF(font).horizontalAdvance(text)
        if tw > bw * 1.18:
            font = _font(max(5.0, fixed_px * max(0.6, (bw * 1.18) / tw)), bold)
            tw = QFontMetricsF(font).horizontalAdvance(text)
    else:
        font, tw = _fit_font(text, bw, bh, bold)
    p.save()
    p.setFont(font)
    p.setPen(QPen(color))
    sx = 1.0
    if stretch and 0 < tw < bw:
        sx = min(bw / tw, 1.10)   # approximate justified lines, capped to avoid distortion
    if align == "center":
        x = x1 + (bw - tw * sx) / 2.0
    elif align == "right":
        x = x2 - tw * sx
    else:
        x = x1
    p.translate(x, y1)
    p.scale(sx, 1.0)
    p.drawText(QRectF(0, 0, max(tw, bw) + 4, bh),
               int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextSingleLine),
               text)
    p.restore()


def _wrapped_height(text: str, font: QFont, width: float) -> float:
    fm = QFontMetricsF(font)
    return fm.boundingRect(QRectF(0, 0, width, 100000.0), int(Qt.TextFlag.TextWordWrap), text).height()


def _draw_wrapped(p: QPainter, text: str, box: List[float], bold: bool, color: QColor, start_px: float):
    """Wrap text inside box, shrinking the font until it fits."""
    text = clean_display_text(text)
    if not text:
        return
    x1, y1, x2, y2 = box
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    px = start_px
    font = _font(px, bold)
    while px > 6 and _wrapped_height(text, font, bw) > bh:
        px *= 0.93
        font = _font(px, bold)
    p.save()
    p.setFont(font)
    p.setPen(QPen(color))
    p.drawText(QRectF(x1, y1, bw, bh + 2),
               int(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap), text)
    p.restore()


def _table_cells(el: Dict[str, Any]) -> Tuple[List[str], List[List[str]]]:
    headers = list(el.get("headers") or [])
    rows = [list(r) for r in (el.get("rows") or [])]
    if not headers and el.get("markdown"):
        import re
        md = [l.strip() for l in el["markdown"].splitlines() if l.strip().startswith("|")]
        parsed = []
        for line in md:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(re.match(r"^:?-+:?$", c) for c in cells):
                continue
            parsed.append(cells)
        if parsed:
            headers, rows = parsed[0], parsed[1:]
    return headers, rows


def _draw_table(p: QPainter, el: Dict[str, Any], box: List[float], line_w: float, body_px: float = 12.0):
    headers, rows = _table_cells(el)
    if not headers:
        return
    x1, y1, x2, y2 = box
    ncols = max(len(headers), max((len(r) for r in rows), default=0), 1)
    nrows = len(rows) + 1
    cw = (x2 - x1) / ncols
    rh = (y2 - y1) / nrows

    p.save()
    p.setBrush(QBrush(COLOR_TABLE_HEAD_BG))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRect(QRectF(x1, y1, x2 - x1, rh))
    for r_idx in range(len(rows)):
        if r_idx % 2 == 1:
            p.setBrush(QBrush(COLOR_TABLE_ALT_BG))
            p.drawRect(QRectF(x1, y1 + (r_idx + 1) * rh, x2 - x1, rh))

    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(COLOR_TABLE_LINE, line_w))
    p.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
    for c in range(1, ncols):
        p.drawLine(int(x1 + c * cw), int(y1), int(x1 + c * cw), int(y2))
    for r in range(1, nrows):
        p.drawLine(int(x1), int(y1 + r * rh), int(x2), int(y1 + r * rh))
    p.restore()

    cell_px = min(rh * 0.55, body_px * 1.15)

    def cell(txt, c, r, bold):
        box_c = [x1 + c * cw + 2, y1 + r * rh + 1, x1 + (c + 1) * cw - 2, y1 + (r + 1) * rh - 1]
        _draw_line(p, str(txt), box_c, bold, COLOR_BODY, align="center", fixed_px=cell_px)

    for c, h in enumerate(headers[:ncols]):
        cell(h, c, 0, True)
    for r_idx, row in enumerate(rows):
        for c, v in enumerate(row[:ncols]):
            cell(v, c, r_idx + 1, False)


def layout_page(elements: List[Dict[str, Any]], W: int, H: int) -> Tuple[List[Tuple[Dict[str, Any], List[float]]], int]:
    """
    Assign a box to every element. Elements that carry coordinates keep them;
    others flow top-to-bottom below the last placed element.
    Returns (items, page_height) where page_height >= H.
    """
    items = []
    margin = W * 0.06
    flow_y = H * 0.03
    body_px = max(10.0, W * 0.018)

    # Collect all image bounding boxes to suppress overlapping text (e.g. chart axis labels / data points)
    image_boxes = []
    for el in elements or []:
        if el.get("type") in ("Image", "Figure", "Picture"):
            b = el.get("box")
            if b and len(b) >= 4 and b[2] > b[0] and b[3] > b[1]:
                image_boxes.append([float(v) for v in b[:4]])

    def sort_key(e):
        b = e.get("box")
        return b[1] if b else e.get("top_y", 0)

    for el in sorted(elements or [], key=sort_key):
        box = el.get("box")
        etype = el.get("type", "Paragraph")

        # Suppress text elements whose boxes collide with an Image (internal chart labels)
        if etype in ("Paragraph", "Heading") and box and len(box) >= 4:
            bx1, by1, bx2, by2 = [float(v) for v in box[:4]]
            collides_with_image = False
            for ib in image_boxes:
                ix1, iy1, ix2, iy2 = ib
                ox = max(0.0, min(bx2, ix2 + 40) - max(bx1, ix1 - 60))
                oy = max(0.0, min(by2, iy2) - max(by1, iy1))
                b_area = max(1.0, (bx2 - bx1) * (by2 - by1))
                if (ox * oy) / b_area > 0.35 or (oy > 0.60 * (by2 - by1) and ox > 0.40 * (bx2 - bx1)):
                    collides_with_image = True
                    break
            if collides_with_image:
                continue

        if box and len(box) >= 4 and box[2] > box[0] and box[3] > box[1]:
            b = [float(v) for v in box[:4]]
            items.append((el, b))
            flow_y = max(flow_y, b[3] + H * 0.005)
            continue

        text = clean_display_text(el.get("text") or el.get("content", ""))
        width = W - 2 * margin
        if etype == "Title":
            h = body_px * 2.2
        elif etype == "Heading":
            h = body_px * 1.8
        elif etype == "Table":
            headers, rows = _table_cells(el)
            h = body_px * 1.9 * (len(rows) + 1)
        elif etype in ("Image", "Figure", "Picture"):
            path = el.get("image_path")
            raw_cap = el.get("caption") or el.get("content") or ""
            if raw_cap.startswith("![") and "](" in raw_cap:
                m_cap = re.match(r"!\[(.*?)\]", raw_cap)
                raw_cap = m_cap.group(1) if m_cap else ""
            caption = clean_display_text(raw_cap)
            has_cap = bool(caption and not re.match(r"^figure\s*\d*$", caption.lower()))
            cap_h = (body_px * 1.6) if has_cap else 0.0

            if path and os.path.exists(path):
                img = QImage(path)
                if not img.isNull() and img.width() > 0 and img.height() > 0:
                    aspect = img.height() / float(img.width())
                    target_w = min(width * 0.82, float(img.width()))
                    target_w = max(target_w, width * 0.38)
                    target_h = target_w * aspect
                    max_h = H * 0.42
                    if target_h > max_h:
                        target_h = max_h
                        target_w = target_h / aspect
                    x1 = (W - target_w) / 2.0
                    x2 = x1 + target_w
                    total_h = target_h + cap_h + body_px * 0.4
                    items.append((el, [x1, flow_y, x2, flow_y + total_h]))
                    flow_y += total_h + body_px * 0.8
                    continue

            h = W * 0.28 + cap_h
            items.append((el, [margin, flow_y, W - margin, flow_y + h]))
            flow_y += h + body_px * 0.8
            continue
        else:
            h = _wrapped_height(text, _font(body_px), width) + body_px * 0.6
        items.append((el, [margin, flow_y, W - margin, flow_y + h]))
        flow_y += h + body_px * 0.6

    page_h = int(max(H, flow_y + H * 0.03))
    return items, page_h


def paint_page(p: QPainter, elements: List[Dict[str, Any]], W: int, H: int) -> int:
    """Paint one page in image coordinates (W x H). Returns the final page height."""
    items, page_h = layout_page(elements, W, H)
    line_w = max(1.0, W * 0.0016)
    body_px = max(10.0, W * 0.018)

    def _median_px(kinds, ratio, default):
        hs = []
        for e, _ in items:
            if e.get("type") in kinds:
                for seg in e.get("lines") or []:
                    hs.append(seg["box"][3] - seg["box"][1])
        if not hs:
            return default
        hs.sort()
        return max(6.0, hs[len(hs) // 2] * ratio)

    para_px = _median_px(("Paragraph",), 0.74, body_px)
    head_px = _median_px(("Heading", "Title"), 0.80, para_px * 1.1)
    body_px = para_px

    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    p.fillRect(QRectF(0, 0, W, page_h), QColor("#ffffff"))

    for el, box in items:
        etype = el.get("type", "Paragraph")
        lines = el.get("lines") or []

        if etype == "Title":
            if lines:
                # Running header: each piece keeps its own position (left / centre / right)
                for seg in lines:
                    _draw_line(p, seg["text"], [float(v) for v in seg["box"]], True, COLOR_TITLE, fixed_px=head_px)
            else:
                _draw_line(p, el.get("text") or el.get("content", ""), box, True, COLOR_TITLE, align="center")
            p.save()
            p.setPen(QPen(QColor("#7fb5d6"), line_w))
            p.drawLine(int(box[0]), int(box[3] + line_w * 2), int(box[2]), int(box[3] + line_w * 2))
            p.restore()

        elif etype == "Heading":
            color = COLOR_H2 if el.get("level", 3) == 2 else COLOR_H3
            if lines:
                for seg in lines:
                    _draw_line(p, seg["text"], [float(v) for v in seg["box"]], True, color, fixed_px=head_px)
            else:
                _draw_line(p, el.get("text") or el.get("content", ""), box, True, color)

        elif etype == "Table":
            _draw_table(p, el, box, line_w, body_px)

        elif etype in ("Image", "Figure", "Picture"):
            path = el.get("image_path")
            raw_cap = el.get("caption") or el.get("content") or ""
            if raw_cap.startswith("![") and "](" in raw_cap:
                m_cap = re.match(r"!\[(.*?)\]", raw_cap)
                raw_cap = m_cap.group(1) if m_cap else ""
            caption = clean_display_text(raw_cap)
            has_caption = bool(caption and not re.match(r"^figure\s*\d*$", caption.lower()))

            caption_h = (body_px * 1.5) if has_caption else 0.0
            avail_w = max(1.0, box[2] - box[0])
            avail_h = max(1.0, (box[3] - box[1]) - caption_h)

            if path and os.path.exists(path):
                img = QImage(path)
                if not img.isNull() and img.width() > 0 and img.height() > 0:
                    img_w, img_h = float(img.width()), float(img.height())
                    # Preserve exact proportional aspect ratio
                    scale = min(avail_w / img_w, avail_h / img_h, 1.0)
                    draw_w = img_w * scale
                    draw_h = img_h * scale
                    draw_x = box[0] + (avail_w - draw_w) / 2.0
                    draw_y = box[1] + (avail_h - draw_h) / 2.0

                    target = QRectF(draw_x, draw_y, draw_w, draw_h)
                    p.save()
                    p.drawImage(target, img)
                    p.setPen(QPen(QColor("#cbd5e1"), max(1.0, line_w * 0.8)))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawRect(target)
                    p.restore()

                    if has_caption:
                        cap_box = [box[0], draw_y + draw_h + body_px * 0.25, box[2], draw_y + draw_h + body_px * 0.25 + caption_h]
                        font_cap = _font(max(7.0, body_px * 0.88), bold=False)
                        font_cap.setItalic(True)
                        p.save()
                        p.setFont(font_cap)
                        p.setPen(QPen(QColor("#475569")))  # Slate gray for figure captions
                        p.drawText(QRectF(cap_box[0], cap_box[1], avail_w, caption_h),
                                   int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
                                   caption)
                        p.restore()

        else:  # Paragraph
            if lines:
                for i, seg in enumerate(lines):
                    # Justify full lines only; the last line of a paragraph stays natural width
                    _draw_line(p, seg["text"], [float(v) for v in seg["box"]], False, COLOR_BODY,
                               stretch=(i < len(lines) - 1), fixed_px=para_px)
            else:
                _draw_wrapped(p, el.get("text") or el.get("content", ""), box, False, COLOR_BODY, body_px)
    p.restore()
    return page_h


def _page_dims(page: Dict[str, Any]) -> Tuple[int, int]:
    shape = page.get("image_shape")
    if not shape and page.get("image") is not None and hasattr(page["image"], "shape"):
        shape = page["image"].shape[:2]
    if not shape:
        return 800, 1100
    return int(shape[1]), int(shape[0])


def render_page_pixmap(page: Dict[str, Any], target_width: int = 800) -> QPixmap:
    """Render one page for on-screen display."""
    W, H = _page_dims(page)
    elements = page.get("elements", [])
    _, page_h = layout_page(elements, W, H)
    scale = target_width / float(W)
    img = QImage(int(W * scale), int(page_h * scale), QImage.Format.Format_RGB32)
    img.fill(QColor("#ffffff"))
    p = QPainter(img)
    p.scale(scale, scale)
    paint_page(p, elements, W, H)
    p.end()
    return QPixmap.fromImage(img)


def export_pdf(pages: List[Dict[str, Any]], output_path: str, title: str = "Digitized Document") -> bool:
    """Write all pages to a PDF; text stays real, selectable text."""
    if not pages:
        return False
    writer = QPdfWriter(output_path)
    writer.setResolution(72)
    writer.setTitle(title)
    writer.setCreator("LiveDoc Layout Renderer")
    writer.setPageMargins(QMarginsF(0, 0, 0, 0))

    painter = None
    try:
        for idx, page in enumerate(pages):
            W, H = _page_dims(page)
            elements = page.get("elements", [])
            _, page_h = layout_page(elements, W, H)
            pw = PDF_PAGE_WIDTH_PT
            ph = pw * page_h / float(W)
            writer.setPageSize(QPageSize(QSizeF(pw, ph), QPageSize.Unit.Point))
            if painter is None:
                painter = QPainter(writer)
                if not painter.isActive():
                    return False
            else:
                writer.newPage()
            painter.save()
            painter.scale(pw / float(W), ph / float(page_h))
            paint_page(painter, elements, W, H)
            painter.restore()
        return True
    finally:
        if painter is not None:
            painter.end()
