"""
Small reusable Qt widgets that make the LiveDoc UI responsive.

- FlowLayout:  lays widgets out left-to-right and wraps onto new rows when the
               available width runs out (used for the toolbar).
- ElidedLabel: a QLabel that shrinks and shows "…" instead of forcing its
               parent window to stay wide (used for the status badges).
"""

from PyQt6.QtCore import Qt, QPoint, QRect, QSize
from PyQt6.QtWidgets import QLabel, QLayout, QSizePolicy


class FlowLayout(QLayout):
    def __init__(self, parent=None, h_spacing: int = 8, v_spacing: int = 8):
        super().__init__(parent)
        self._items = []
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing
        self._right_align = False
        self.setContentsMargins(0, 0, 0, 0)

    # ── QLayout interface ────────────────────────────────
    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        """Size needed to show every item on a single row."""
        width = height = 0
        count = 0
        for item in self._items:
            if item.isEmpty():
                continue
            hint = item.sizeHint()
            width += hint.width()
            height = max(height, hint.height())
            count += 1
        if count > 1:
            width += self._h_spacing * (count - 1)
        m = self.contentsMargins()
        return QSize(width + m.left() + m.right(), height + m.top() + m.bottom())

    def minimumSize(self):
        """Narrowest layout: one item per row."""
        width = height = 0
        for item in self._items:
            if item.isEmpty():
                continue
            hint = item.minimumSize()
            width = max(width, hint.width())
            height = max(height, hint.height())
        m = self.contentsMargins()
        return QSize(width + m.left() + m.right(), height + m.top() + m.bottom())

    # ── Extras ───────────────────────────────────────────
    def set_right_align(self, right_align: bool):
        if self._right_align != right_align:
            self._right_align = right_align
            self.invalidate()

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        max_w = max(area.width(), 1)

        rows = []  # each row: ([(item, size), ...], row_width, row_height)
        row, row_w, row_h = [], 0, 0
        for item in self._items:
            if item.isEmpty():
                continue
            hint = item.sizeHint()
            size = QSize(min(hint.width(), max_w), hint.height())
            needed = row_w + (self._h_spacing if row else 0) + size.width()
            if row and needed > max_w:
                rows.append((row, row_w, row_h))
                row, row_w, row_h = [], 0, 0
                needed = size.width()
            row.append((item, size))
            row_w = needed
            row_h = max(row_h, size.height())
        if row:
            rows.append((row, row_w, row_h))

        y = area.y()
        for row_items, row_width, row_height in rows:
            x = area.x()
            if self._right_align:
                x += max(0, area.width() - row_width)
            for item, size in row_items:
                if not test_only:
                    offset = (row_height - size.height()) // 2
                    item.setGeometry(QRect(QPoint(x, y + offset), size))
                x += size.width() + self._h_spacing
            y += row_height + self._v_spacing

        content_height = (y - self._v_spacing - area.y()) if rows else 0
        return content_height + m.top() + m.bottom()


class ElidedLabel(QLabel):
    """QLabel that elides overflowing text and never demands extra width."""

    MIN_WIDTH = 90

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._full_text = ""
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setText(text)

    def setText(self, text):
        self._full_text = text
        self.setToolTip(text.strip())
        self._refresh()

    def fullText(self) -> str:
        return self._full_text

    def sizeHint(self):
        base = super().sizeHint()
        fm = self.fontMetrics()
        chrome = base.width() - fm.horizontalAdvance(self.text())
        return QSize(fm.horizontalAdvance(self._full_text) + chrome, base.height())

    def minimumSizeHint(self):
        return QSize(self.MIN_WIDTH, super().minimumSizeHint().height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self):
        available = self.contentsRect().width()
        fm = self.fontMetrics()
        if available <= 0 or fm.horizontalAdvance(self._full_text) <= available:
            shown = self._full_text
        else:
            shown = fm.elidedText(self._full_text, Qt.TextElideMode.ElideRight, available)
        if shown != self.text():
            super().setText(shown)
