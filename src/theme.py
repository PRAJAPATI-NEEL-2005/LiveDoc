"""
LiveDoc theming: light / dark palettes, Qt stylesheet builders, persisted
theme preference and HTML colour adaptation for the formatted-document view.
"""

import json
import os
import re
from string import Template

CONFIG_FILE_PATH = os.path.expanduser("~/.livedoc_config.json")

DARK = "dark"
LIGHT = "light"

_FONT_STACK = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif"

# Saturated action buttons look good with white text in both themes.
_BUTTONS = {
    "btn_camera": "#1d4ed8", "btn_camera_hover": "#2563eb",
    "btn_camera_stop": "#b91c1c", "btn_camera_stop_hover": "#dc2626",
    "btn_scan": "#059669", "btn_scan_hover": "#10b981",
    "btn_export": "#0284c7", "btn_export_hover": "#0369a1",
    "btn_upload": "#7c3aed", "btn_upload_hover": "#6d28d9",
    "btn_pdf": "#dc2626", "btn_pdf_hover": "#b91c1c",
}

PALETTES = {
    DARK: {
        **_BUTTONS,
        "bg": "#0d1117",
        "surface": "#161b22",
        "surface_alt": "#21262d",
        "surface_hover": "#30363d",
        "border": "#30363d",
        "text": "#e2e8f0",
        "text_muted": "#94a3b8",
        "text_subtle": "#8b949e",
        "accent": "#38bdf8",
        "accent_strong": "#1e40af",
        "selection": "#1e40af",
        "cam_bg": "#161b22",
        "status_bg": "#0a0d14",
        "status_border": "#21262d",
        "scroll_handle": "#30363d",
        "scroll_handle_hover": "#484f58",
        "btn_scan_disabled_bg": "#1f2937",
        "btn_scan_disabled_fg": "#4b5563",
        "badge_ok_bg": "#064e3b", "badge_ok_fg": "#6ee7b7",
        "badge_bad_bg": "#7f1d1d", "badge_bad_fg": "#fca5a5",
        "badge_warn_bg": "#78350f", "badge_warn_fg": "#fcd34d",
        "badge_ai_bg": "#082f49", "badge_ai_fg": "#38bdf8", "badge_ai_border": "#0284c7",
        "ok": "#6ee7b7",
        "err": "#fca5a5",
        "error_text": "#ef4444",
    },
    LIGHT: {
        **_BUTTONS,
        "bg": "#f1f5f9",
        "surface": "#ffffff",
        "surface_alt": "#e2e8f0",
        "surface_hover": "#cbd5e1",
        "border": "#cbd5e1",
        "text": "#0f172a",
        "text_muted": "#475569",
        "text_subtle": "#64748b",
        "accent": "#0369a1",
        "accent_strong": "#2563eb",
        "selection": "#bfdbfe",
        "cam_bg": "#e2e8f0",
        "status_bg": "#e2e8f0",
        "status_border": "#cbd5e1",
        "scroll_handle": "#cbd5e1",
        "scroll_handle_hover": "#94a3b8",
        "btn_scan_disabled_bg": "#e2e8f0",
        "btn_scan_disabled_fg": "#94a3b8",
        "badge_ok_bg": "#d1fae5", "badge_ok_fg": "#047857",
        "badge_bad_bg": "#fee2e2", "badge_bad_fg": "#b91c1c",
        "badge_warn_bg": "#fef3c7", "badge_warn_fg": "#92400e",
        "badge_ai_bg": "#e0f2fe", "badge_ai_fg": "#0369a1", "badge_ai_border": "#0284c7",
        "ok": "#047857",
        "err": "#b91c1c",
        "error_text": "#dc2626",
    },
}


# ─────────────────────────────────────────────────────────
# Preference persistence (shares ~/.livedoc_config.json with the API key)
# ─────────────────────────────────────────────────────────
def load_saved_theme(default: str = DARK) -> str:
    try:
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            value = json.load(f).get("theme", default)
        return value if value in PALETTES else default
    except Exception:
        return default


def save_theme(name: str) -> None:
    cfg = {}
    try:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
    except Exception:
        cfg = {}
    cfg["theme"] = name
    try:
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"Warning: could not save theme preference: {e}")


# ─────────────────────────────────────────────────────────
# Stylesheets
# ─────────────────────────────────────────────────────────
_APP_QSS = Template("""
QMainWindow, QWidget {
    background-color: $bg;
    color: $text;
    font-family: $font;
    font-size: 13px;
}
QLabel { background: transparent; }
QToolTip {
    background-color: $surface;
    color: $text;
    border: 1px solid $border;
    padding: 4px 8px;
}

QLabel#title_label {
    font-size: 22px;
    font-weight: 800;
    color: $accent;
    padding: 4px 0px;
}
QLabel#subtitle_label {
    font-size: 12px;
    color: $text_muted;
    padding-bottom: 6px;
}
QLabel#cam_feed {
    background-color: $cam_bg;
    color: $text_muted;
    border: 2px solid $border;
    border-radius: 12px;
    font-size: 14px;
}
QLabel#section_header {
    font-size: 12px;
    font-weight: bold;
    color: $text_muted;
    letter-spacing: 1px;
    padding: 4px 0px;
}
QLabel#ctrl_label { color: $text_muted; font-weight: bold; }
QLabel#ctrl_label_accent { color: $accent; font-weight: bold; }
QLabel#visual_placeholder { color: $text_muted; font-size: 13px; padding: 25px; }
QLabel#visual_page { padding: 12px; }

QScrollArea#visual_scroll { background-color: $surface; border: none; }
QScrollArea#visual_scroll QWidget#qt_scrollarea_viewport { background-color: $surface; }

QSplitter::handle { background-color: $border; }

QTabWidget::pane {
    border: 2px solid $border;
    border-radius: 8px;
    background-color: $surface;
}
QTabBar::tab {
    background: $bg;
    color: $text_muted;
    font-weight: bold;
    padding: 8px 14px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 4px;
}
QTabBar::tab:selected {
    background: $surface;
    color: $accent;
    border-bottom: 2px solid $accent;
}
QTabBar::tab:hover { color: $text; }

QTextEdit, QTextBrowser {
    background-color: $surface;
    border: none;
    color: $text;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 13px;
    padding: 14px;
    selection-background-color: $selection;
    selection-color: $text;
}

QPushButton {
    background-color: $surface_alt;
    color: $text;
    border: 1px solid $border;
    border-radius: 8px;
    padding: 10px 18px;
    font-size: 13px;
    font-weight: 600;
}
QPushButton:hover { background-color: $surface_hover; }
QPushButton:disabled { color: $text_subtle; }
QPushButton#btn_theme {
    padding: 6px 14px;
    border-radius: 16px;
}
QPushButton#btn_camera {
    background-color: $btn_camera;
    color: #ffffff;
    border: 1px solid transparent;
}
QPushButton#btn_camera:hover { background-color: $btn_camera_hover; }
QPushButton#btn_camera:checked { background-color: $btn_camera_stop; }
QPushButton#btn_camera:checked:hover { background-color: $btn_camera_stop_hover; }
QPushButton#btn_scan {
    background-color: $btn_scan;
    color: #ffffff;
    border: 1px solid transparent;
    font-size: 14px;
    font-weight: bold;
    padding: 12px 24px;
}
QPushButton#btn_scan:hover { background-color: $btn_scan_hover; }
QPushButton#btn_scan:disabled {
    background-color: $btn_scan_disabled_bg;
    color: $btn_scan_disabled_fg;
}
QPushButton#btn_export {
    background-color: $btn_export;
    color: #ffffff;
    border: 1px solid transparent;
}
QPushButton#btn_export:hover { background-color: $btn_export_hover; }
QPushButton#btn_upload {
    background-color: $btn_upload;
    color: #ffffff;
    border: 1px solid transparent;
    font-weight: bold;
}
QPushButton#btn_upload:hover { background-color: $btn_upload_hover; }
QPushButton#btn_pdf {
    background-color: $btn_pdf;
    color: #ffffff;
    border: 1px solid transparent;
    font-weight: bold;
}
QPushButton#btn_pdf:hover { background-color: $btn_pdf_hover; }

QStatusBar {
    background-color: $status_bg;
    color: $text_subtle;
    font-size: 12px;
    padding: 3px 10px;
    border-top: 1px solid $status_border;
}
QProgressBar {
    background-color: $surface_alt;
    border-radius: 4px;
    border: none;
    max-height: 4px;
}
QProgressBar::chunk {
    background-color: $accent;
    border-radius: 4px;
}

QComboBox {
    background-color: $surface;
    color: $text;
    border: 1px solid $border;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 13px;
    min-width: 120px;
}
QComboBox:hover { border-color: $accent; }
QComboBox QAbstractItemView {
    background-color: $surface;
    color: $text;
    border: 1px solid $border;
    selection-background-color: $selection;
    selection-color: $text;
    outline: 0;
}

QScrollBar:vertical { background: transparent; width: 12px; margin: 0; }
QScrollBar::handle:vertical {
    background: $scroll_handle;
    border-radius: 5px;
    min-height: 28px;
    margin: 2px;
}
QScrollBar::handle:vertical:hover { background: $scroll_handle_hover; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 0; }
QScrollBar::handle:horizontal {
    background: $scroll_handle;
    border-radius: 5px;
    min-width: 28px;
    margin: 2px;
}
QScrollBar::handle:horizontal:hover { background: $scroll_handle_hover; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0px; height: 0px; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

QLabel#badge_searching, QLabel#badge_found, QLabel#badge_blurry,
QLabel#badge_confidence_neutral, QLabel#badge_confidence_high,
QLabel#badge_confidence_fair, QLabel#badge_confidence_smart_ai {
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: bold;
}
QLabel#badge_searching, QLabel#badge_confidence_neutral {
    background-color: $surface_alt;
    color: $text_subtle;
}
QLabel#badge_found, QLabel#badge_confidence_high {
    background-color: $badge_ok_bg;
    color: $badge_ok_fg;
}
QLabel#badge_blurry {
    background-color: $badge_bad_bg;
    color: $badge_bad_fg;
}
QLabel#badge_confidence_fair {
    background-color: $badge_warn_bg;
    color: $badge_warn_fg;
}
QLabel#badge_confidence_smart_ai {
    background-color: $badge_ai_bg;
    color: $badge_ai_fg;
    border: 1px solid $badge_ai_border;
}
""")

_DIALOG_QSS = Template("""
QDialog {
    background-color: $bg;
    color: $text;
    font-family: $font;
}
QLabel { color: $text; font-size: 13px; background: transparent; }
QLabel#dlg_title { font-size: 16px; font-weight: bold; color: $accent; }
QLabel#dlg_muted { color: $text_muted; font-size: 12px; }
QLabel#dlg_label { font-weight: bold; color: $text; }
QLabel#dlg_status_ok { color: $ok; font-weight: bold; font-size: 12px; }
QLabel#dlg_status_err { color: $err; font-weight: bold; font-size: 12px; }
QFrame#dlg_info {
    background-color: $surface;
    border: 1px solid $border;
    border-radius: 8px;
}
QLineEdit {
    background-color: $surface;
    border: 1px solid $border;
    border-radius: 6px;
    color: $text;
    padding: 8px 12px;
    font-family: monospace;
    font-size: 13px;
}
QLineEdit:focus { border-color: $accent; }
QCheckBox { color: $text_muted; font-size: 12px; }
QPushButton {
    background-color: $surface_alt;
    color: $text;
    border: 1px solid $border;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 600;
}
QPushButton:hover { background-color: $surface_hover; }
QPushButton#dlg_primary {
    background-color: $btn_export;
    color: #ffffff;
    border: 1px solid transparent;
}
QPushButton#dlg_primary:hover { background-color: $btn_export_hover; }
""")


def build_app_stylesheet(name: str) -> str:
    return _APP_QSS.substitute(PALETTES[name], font=_FONT_STACK)


def build_dialog_stylesheet(name: str) -> str:
    return _DIALOG_QSS.substitute(PALETTES[name], font=_FONT_STACK)


# ─────────────────────────────────────────────────────────
# HTML adaptation for the "Formatted Document" tab
#
# The offline engine emits HTML authored for a dark background, while the
# Smart AI engine reuses the print HTML authored for white paper. Each rule
# set below remaps the hard-coded inline colours to suit the active theme.
# Rules are applied in a single pass so one replacement never feeds another.
# ─────────────────────────────────────────────────────────
SOURCE_DARK = "dark"    # HTML authored for a dark background (offline engine)
SOURCE_PRINT = "print"  # HTML authored for white paper (Smart AI engine)

_DARK_SOURCE_TO_LIGHT = [
    (r"color:\s*#e2e8f0", "color: #1e293b"),
    (r"#38bdf8", "#0369a1"),
    (r"#67e8f9", "#0e7490"),
    (r"background:\s*#1e293b", "background: #e2e8f0"),
    (r"background-color:\s*#1e293b", "background-color: #f1f5f9"),
    (r"background-color:\s*#0f172a", "background-color: #ffffff"),
    (r"color:\s*#94a3b8", "color: #475569"),
    (r"color:\s*#f8fafc", "color: #0f172a"),
    (r"solid\s+#334155", "solid #cbd5e1"),
]

_PRINT_SOURCE_TO_DARK = [
    (r"color:\s*#0f172a", "color: #e2e8f0"),
    (r"color:\s*#1e293b", "color: #e2e8f0"),
    (r"color:\s*#334155", "color: #cbd5e1"),
    (r"color:\s*#475569", "color: #94a3b8"),
    (r"color:\s*#64748b", "color: #94a3b8"),
    (r"color:\s*#0284c7", "color: #38bdf8"),
    (r"color:\s*#0369a1", "color: #38bdf8"),
    (r"solid\s+#334155", "solid #475569"),
    (r"solid\s+#cbd5e1", "solid #30363d"),
    (r"solid\s+#e2e8f0", "solid #30363d"),
    (r"solid\s+#0284c7", "solid #38bdf8"),
    (r"background-color:\s*#f1f5f9", "background-color: #1e293b"),
    (r'(?<=<tr style=")background-color:\s*#ffffff', "background-color: #161b22"),
    (r"background-color:\s*#f8fafc", "background-color: #1c2128"),
]


def _compile_rules(rules):
    combined = re.compile(
        "|".join(f"(?P<r{i}>{pattern})" for i, (pattern, _) in enumerate(rules)),
        re.IGNORECASE,
    )

    def apply(html: str) -> str:
        def replace(match):
            for i, (_, replacement) in enumerate(rules):
                if match.group(f"r{i}") is not None:
                    return replacement
            return match.group(0)
        return combined.sub(replace, html)

    return apply


_to_light = _compile_rules(_DARK_SOURCE_TO_LIGHT)
_to_dark = _compile_rules(_PRINT_SOURCE_TO_DARK)


def adapt_html(html: str, source: str, theme_name: str) -> str:
    """Re-colour backend-generated HTML so it is readable in `theme_name`."""
    if not html:
        return html
    if source == SOURCE_DARK and theme_name == LIGHT:
        return _to_light(html)
    if source == SOURCE_PRINT and theme_name == DARK:
        return _to_dark(html)
    return html
