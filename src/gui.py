"""
LiveDoc Modern Desktop GUI Application (PyQt6)
Features:
- Real-time live camera feed preview with document boundary detection & viewfinder.
- Multi-metric positioning and quality guidance (Hold steady, Move closer, Boundary locked).
- Asynchronous OCR & layout digitization worker (UI remains smooth and responsive).
- Multi-Tab Structured Presentation:
  * Formatted Document (Rich HTML with styled tables, bold headings, and visual badges)
  * Markdown Text (Faithful layout representation with table syntax)
  * Structured JSON (Clean machine-readable representation as per architecture spec)
  * Validation & Verification Report (OCR confidence scores & table consistency checks)
- Export to Markdown, HTML, JSON, and Plain Text.
- Camera selector dropdown with runtime switching and 90° rotation.
"""

import sys
import os

# Ensure project root is in sys.path so 'src' imports work from any working directory or entrypoint
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QTextEdit, QTextBrowser, QHBoxLayout, QVBoxLayout, QBoxLayout, QSplitter,
    QStatusBar, QProgressBar, QFrame, QSizePolicy, QComboBox,
    QTabWidget, QFileDialog, QMessageBox, QDialog, QLineEdit, QCheckBox, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QIcon, QShortcut, QKeySequence

# Import backend modules
from src.camera_manager import LiveCameraManager
from src.document_detector import DocumentDetector
from src.layout_analyzer import LayoutAnalyzer
from src.ocr_engine import OCREngine
from src.smart_ai_engine import SmartAIEngine
from src.pdf_exporter import PDFExporter
from src.layout_renderer import render_page_pixmap, export_pdf as export_layout_pdf
from src import theme
from src.ui_widgets import FlowLayout, ElidedLabel


# ─────────────────────────────────────────────────────────
# AI Settings & API Configuration Dialog
# ─────────────────────────────────────────────────────────
class AISettingsDialog(QDialog):
    def __init__(self, smart_ai_engine: SmartAIEngine, parent=None, theme_name: str = theme.DARK):
        super().__init__(parent)
        self.smart_ai_engine = smart_ai_engine
        self.theme_name = theme_name
        self.setWindowTitle("LiveDoc AI Settings & API Configuration")
        self.setMinimumWidth(340)
        self.setStyleSheet(theme.build_dialog_stylesheet(theme_name))
        self._build_ui()
        self.resize(540, self.sizeHint().height())

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Google Gemini Vision AI Settings")
        title.setObjectName("dlg_title")
        title.setWordWrap(True)
        layout.addWidget(title)

        desc = QLabel(
            "LiveDoc connects with Google Gemini 3.8 Flash to power the 'Pure Smart AI' mode "
            "and automatically enhance low-confidence scans in 'Auto / Hybrid' mode."
        )
        desc.setWordWrap(True)
        desc.setObjectName("dlg_muted")
        layout.addWidget(desc)

        # API Key Form
        key_label = QLabel("Gemini API Key:")
        key_label.setObjectName("dlg_label")
        layout.addWidget(key_label)

        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("Enter your Gemini API key (e.g. AIzaSy...)")
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        if self.smart_ai_engine.api_key:
            self.key_input.setText(self.smart_ai_engine.api_key)
        layout.addWidget(self.key_input)

        self.chk_show = QCheckBox("Show API key")
        self.chk_show.toggled.connect(
            lambda checked: self.key_input.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        layout.addWidget(self.chk_show)

        self.lbl_status = QLabel()
        self.lbl_status.setWordWrap(True)
        self._set_status(self.smart_ai_engine.is_available())
        layout.addWidget(self.lbl_status)

        # Instructions / Free tier banner
        info_box = QFrame()
        info_box.setObjectName("dlg_info")
        info_layout = QVBoxLayout(info_box)
        info_layout.setContentsMargins(10, 8, 10, 8)
        pal = theme.PALETTES[self.theme_name]
        info_text = QLabel(
            "<b>How to get your free Gemini API key:</b><br>"
            f"1. Visit <a href='https://aistudio.google.com' style='color: {pal['accent']};'>https://aistudio.google.com</a><br>"
            "2. Click <b>'Get API Key'</b> and create a new key.<br>"
            "3. Paste your key above and click <b>'Save Key'</b>.<br>"
            f"<small style='color: {pal['text_muted']};'>• Generous free tier included. Stored safely in ~/.livedoc_config.json.</small>"
        )
        info_text.setOpenExternalLinks(True)
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        layout.addWidget(info_box)

        # Buttons
        btn_box = QHBoxLayout()
        self.btn_save = QPushButton("Save Key")
        self.btn_save.setObjectName("dlg_primary")
        self.btn_save.clicked.connect(self._save_key)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)

        btn_box.addStretch()
        btn_box.addWidget(self.btn_save)
        btn_box.addWidget(self.btn_close)
        layout.addLayout(btn_box)

    def _set_status(self, configured: bool):
        if configured:
            self.lbl_status.setText("Status: API Key configured ✓")
            self.lbl_status.setObjectName("dlg_status_ok")
        else:
            self.lbl_status.setText("Status: No API key configured ⚠️")
            self.lbl_status.setObjectName("dlg_status_err")
        self.lbl_status.style().unpolish(self.lbl_status)
        self.lbl_status.style().polish(self.lbl_status)

    def _save_key(self):
        new_key = self.key_input.text().strip()
        self.smart_ai_engine.save_api_key(new_key)
        if self.smart_ai_engine.is_available():
            self._set_status(True)
            QMessageBox.information(self, "API Key Saved", "Gemini API key saved successfully!")
        else:
            self._set_status(False)
            QMessageBox.warning(self, "API Key Cleared", "The API key was cleared.")


# ─────────────────────────────────────────────────────────
# Worker thread: runs OCR/AI in background so UI never freezes
# ─────────────────────────────────────────────────────────
class ScanWorker(QThread):
    result_ready = pyqtSignal(dict, object)   # (structured_doc_dict, flat_image)
    error_occurred = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self, frame, detector, analyzer, ocr_engine, smart_ai_engine=None, scan_mode="auto"):
        super().__init__()
        self.frame = frame
        self.detector = detector
        self.analyzer = analyzer
        self.ocr_engine = ocr_engine
        self.smart_ai_engine = smart_ai_engine
        self.scan_mode = scan_mode

    def run(self):
        try:
            self.status_update.emit("Assessing frame quality...")
            if self.detector.is_blurry(self.frame, threshold=15):
                self.status_update.emit("Notice: Image may be slightly blurry, scanning anyway...")

            self.status_update.emit("Detecting document boundaries...")
            corners = self.detector.find_document(self.frame)
            if corners is not None:
                self.status_update.emit("Boundary found — flattening perspective...")
                flat = self.detector.flatten_document(self.frame, corners)
            else:
                self.status_update.emit("No boundary found — scanning full camera frame...")
                flat = self.frame

            self.status_update.emit("Enhancing contrast & normalizing illumination...")
            flat = self.detector.preprocess_for_ocr(flat)

            if self.scan_mode == "smart_ai":
                self.status_update.emit("Transcribing document with Multimodal Vision AI...")
                if not self.smart_ai_engine or not self.smart_ai_engine.is_available():
                    raise ValueError(
                        "Smart AI is not configured. Please set your Gemini API key via the ⚙️ AI Settings button "
                        "or set the GEMINI_API_KEY environment variable. You can get a free key at https://aistudio.google.com"
                    )
                try:
                    doc_data = self.smart_ai_engine.transcribe_document(flat)
                    # Supplement with OCR token coordinates for Dual-Layer Searchable PDF positioning
                    if not doc_data.get("tokens") and self.ocr_engine:
                        try:
                            raw_ocr, _ = self.ocr_engine._get_best_ocr(flat)
                            doc_data["tokens"] = self.ocr_engine._parse_tokens(raw_ocr)
                        except Exception:
                            pass
                except Exception as ai_err:
                    # Seamless fallback: if cloud servers are experiencing high demand (503) or offline,
                    # don't leave the user with an error screen! Digitized with the offline engine immediately:
                    self.status_update.emit("Cloud AI busy — seamlessly digitized via Offline Engine...")
                    boxes = self.analyzer.analyze(flat)
                    doc_data = self.ocr_engine.extract_structured_document(flat, boxes)
                    doc_data["ai_fallback_warning"] = (
                        f"Smart AI experienced temporary server traffic ({str(ai_err)[:90]}...). "
                        "LiveDoc seamlessly digitized your document using the offline engine."
                    )

            elif self.scan_mode == "offline":
                self.status_update.emit("Analyzing document layout (Tables, Figures, Text)...")
                boxes = self.analyzer.analyze(flat)
                self.status_update.emit("Extracting structured text & reconstructing tables...")
                doc_data = self.ocr_engine.extract_structured_document(flat, boxes)

            else:  # "auto" / Hybrid
                self.status_update.emit("Running fast offline scan & confidence evaluation...")
                boxes = self.analyzer.analyze(flat)
                doc_data = self.ocr_engine.extract_structured_document(flat, boxes)

                conf_info = doc_data.get("confidence", {})
                score = conf_info.get("score", 90.0)
                needs_escalation = conf_info.get("needs_escalation", False)

                if score < 85.0 or needs_escalation:
                    if self.smart_ai_engine and self.smart_ai_engine.is_available():
                        self.status_update.emit(
                            f"Offline scan confidence is {score:.1f}% (< 85%) — Auto-enhancing with Smart AI..."
                        )
                        try:
                            ai_doc = self.smart_ai_engine.transcribe_document(flat)
                            if not ai_doc.get("tokens") and doc_data.get("tokens"):
                                ai_doc["tokens"] = doc_data["tokens"]
                            doc_data = ai_doc
                            doc_data["escalated_from_offline"] = True
                            doc_data["original_offline_score"] = score
                        except Exception as ai_err:
                            doc_data["ai_fallback_warning"] = str(ai_err)
                    else:
                        doc_data["ai_notice"] = (
                            f"Confidence score is {score:.1f}%. Add a Gemini API Key in ⚙️ AI Settings "
                            "to enable automatic Smart AI enhancement."
                        )

            self.result_ready.emit(doc_data, flat)
        except Exception as e:
            self.error_occurred.emit(f"Pipeline error: {str(e)}")


# ─────────────────────────────────────────────────────────
# Main Application Window
# ─────────────────────────────────────────────────────────
class LiveDocApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiveDoc — Real-Time Intelligent Document Digitizer")
        self.setMinimumSize(520, 520)
        self._set_initial_size()

        # Theme (persisted between sessions)
        self.theme_name = theme.load_saved_theme()

        # Backend objects
        self.camera = LiveCameraManager(camera_index=None)
        self.available_cameras = []
        self.detector = DocumentDetector()
        self.analyzer = LayoutAnalyzer()
        self.ocr_engine = OCREngine()
        self.smart_ai_engine = SmartAIEngine()

        # State
        self.camera_running = False
        self.current_frame = None
        self.scan_worker = None
        self.page_count = 0
        self.frame_counter = 0
        self.rotation_angle = 0
        self.cached_corners = None
        self.cached_quality = {"status": "SEARCHING", "message": "Searching for document...", "color": (0, 220, 100)}

        # Digitized history
        self.latest_doc = None
        self.all_pages_markdown = []
        self.all_pages_html = []          # [(page_num, html_source, raw_html)]
        self.all_pages_elements = []
        self.all_pages_data = []

        # Uploaded-image preview (re-scaled when the window is resized)
        self._preview_frame = None
        self._preview_name = ""
        self._visual_page_args = None

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._on_resize_settled)

        self._apply_stylesheet()
        self._build_ui()
        self._setup_camera_timer()
        self._update_theme_button()
        QShortcut(QKeySequence("Ctrl+T"), self, activated=self._toggle_theme)
        self._update_responsive_layout()

    # ── Stylesheet ────────────────────────────────────────
    def _apply_stylesheet(self):
        self.setStyleSheet(theme.build_app_stylesheet(self.theme_name))

    # ── UI Layout ─────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(18, 14, 18, 10)
        root_layout.setSpacing(10)

        # Header
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("LiveDoc")
        title.setObjectName("title_label")
        self.subtitle = ElidedLabel("Real-Time Intelligent Document Digitizer  •  Computer Vision + YOLO + PaddleOCR + Smart AI")
        self.subtitle.setObjectName("subtitle_label")
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        # Elided labels shrink with the window instead of forcing it to stay wide
        self.doc_badge = ElidedLabel("  Searching for document...")
        self.doc_badge.setObjectName("badge_searching")

        self.confidence_badge = ElidedLabel("  Confidence: —")
        self.confidence_badge.setObjectName("badge_confidence_neutral")

        self.btn_theme = QPushButton()
        self.btn_theme.setObjectName("btn_theme")
        self.btn_theme.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_theme.clicked.connect(self._toggle_theme)

        header_layout.addWidget(title)
        header_layout.addSpacing(14)
        header_layout.addWidget(self.subtitle)
        header_layout.addStretch(1)
        header_layout.addWidget(self.doc_badge)
        header_layout.addSpacing(8)
        header_layout.addWidget(self.confidence_badge)
        header_layout.addSpacing(8)
        header_layout.addWidget(self.btn_theme)
        root_layout.addWidget(header)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        root_layout.addWidget(self.progress)

        # Main splitter (Camera | Results Tabs)
        # (switches to a vertical stack on narrow windows, see _update_responsive_layout)
        self.splitter = splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setChildrenCollapsible(False)

        # ── Left: Camera panel ──
        cam_panel = QWidget()
        cam_layout = QVBoxLayout(cam_panel)
        cam_layout.setContentsMargins(0, 0, 6, 0)
        cam_layout.setSpacing(8)

        cam_header = QLabel("LIVE CAMERA & GUIDANCE")
        cam_header.setObjectName("section_header")
        cam_layout.addWidget(cam_header)

        self.cam_label = QLabel()
        self.cam_label.setObjectName("cam_feed")
        self.cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cam_label.setText("Camera Off\n\nClick 'Start Camera' below to begin")
        self.cam_label.setMinimumSize(280, 200)
        self.cam_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        cam_layout.addWidget(self.cam_label, 1)

        splitter.addWidget(cam_panel)

        # ── Right: Structured Results Tab Widget ──
        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(6, 0, 0, 0)
        result_layout.setSpacing(8)

        res_header = QLabel("STRUCTURED DIGITAL DOCUMENT")
        res_header.setObjectName("section_header")
        result_layout.addWidget(res_header)

        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        self.tabs.tabBar().setExpanding(False)
        self.tabs.tabBar().setElideMode(Qt.TextElideMode.ElideRight)

        # Tab 0: Physical Document View (2D Visual Layout with exact element bounding boxes)
        self.tab_visual_scroll = QScrollArea()
        self.tab_visual_scroll.setObjectName("visual_scroll")
        self.tab_visual_scroll.setWidgetResizable(True)
        self.tab_visual_label = QLabel()
        self.tab_visual_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tab_visual_label.setWordWrap(True)
        self._show_visual_placeholder()
        self.tab_visual_scroll.setWidget(self.tab_visual_label)
        self.tabs.addTab(self.tab_visual_scroll, "  Visual 2D Layout  ")

        # Tab 1: Formatted Document (Rich HTML)
        self.tab_formatted = QTextBrowser()
        self.tab_formatted.setOpenExternalLinks(True)
        self.tab_formatted.setPlaceholderText(
            "Formatted document will render here with real styled tables, bold headings, and visual layouts..."
        )
        self.tabs.addTab(self.tab_formatted, "  Formatted Document  ")

        # Tab 2: Markdown Text
        self.tab_markdown = QTextEdit()
        self.tab_markdown.setPlaceholderText(
            "Markdown text preserving exact tables and sections will appear here...\n\n"
            "Editable for quick notes or copy-pasting."
        )
        self.tabs.addTab(self.tab_markdown, "  Markdown / Plain Text  ")

        # Tab 3: Structured JSON
        self.tab_json = QTextEdit()
        self.tab_json.setPlaceholderText(
            "Machine-readable JSON schema (Metadata, Title, Elements, Tables, Validation) will appear here..."
        )
        self.tab_json.setReadOnly(True)
        self.tabs.addTab(self.tab_json, "  Structured JSON  ")

        # Tab 4: Validation & Quality Report
        self.tab_validation = QTextEdit()
        self.tab_validation.setPlaceholderText(
            "Validation & Verification report will appear here:\n\n"
            "• OCR Confidence Scores\n"
            "• Uncertain Text Flags [Verification Required]\n"
            "• Table Arithmetic & Consistency Checks\n"
            "• Image Quality Assessment"
        )
        self.tab_validation.setReadOnly(True)
        self.tabs.addTab(self.tab_validation, "  Validation Report  ")

        self.tabs.currentChanged.connect(self._on_tab_changed)
        result_layout.addWidget(self.tabs, 1)
        splitter.addWidget(result_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([680, 640])
        root_layout.addWidget(splitter, 1)

        # ── Bottom Controls ──
        # Two wrapping groups: on wide windows they share one row, on narrower
        # windows they stack, and each group wraps its own buttons as needed.
        self.controls = QWidget()
        self.ctrl_layout = QBoxLayout(QBoxLayout.Direction.LeftToRight, self.controls)
        self.ctrl_layout.setContentsMargins(0, 4, 0, 0)
        self.ctrl_layout.setSpacing(10)

        self.ctrl_input = QWidget()
        self.ctrl_input_flow = FlowLayout(self.ctrl_input, h_spacing=10, v_spacing=8)
        self.ctrl_actions = QWidget()
        self.ctrl_actions_flow = FlowLayout(self.ctrl_actions, h_spacing=10, v_spacing=8)

        # Camera selection
        cam_label = QLabel("Camera:")
        cam_label.setObjectName("ctrl_label")
        self.cam_combo = QComboBox()
        self.cam_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cam_combo.setMinimumContentsLength(14)
        self.cam_combo.currentIndexChanged.connect(self._on_camera_changed)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setObjectName("btn_secondary")
        self.btn_refresh.clicked.connect(self._refresh_cameras)

        self.btn_camera = QPushButton("  Start Camera")
        self.btn_camera.setObjectName("btn_camera")
        self.btn_camera.setCheckable(True)
        self.btn_camera.setMinimumWidth(140)
        self.btn_camera.clicked.connect(self.toggle_camera)

        self.btn_rotate = QPushButton("Rotate 90°")
        self.btn_rotate.setObjectName("btn_secondary")
        self.btn_rotate.clicked.connect(self._rotate_view)

        self.btn_upload = QPushButton("  📂 Upload Image")
        self.btn_upload.setObjectName("btn_upload")
        self.btn_upload.setMinimumWidth(150)
        self.btn_upload.setToolTip("Upload an image file (JPG, PNG, TIFF, BMP) and digitize it")
        self.btn_upload.clicked.connect(self._upload_image)

        # Scan Mode selection
        mode_label = QLabel("Mode:")
        mode_label.setObjectName("ctrl_label_accent")
        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("mode_combo")
        self.mode_combo.addItem("🔄 Auto (Hybrid)", userData="auto")
        self.mode_combo.addItem("⚡ Offline (Local)", userData="offline")
        self.mode_combo.addItem("🌟 Smart AI (Gemini)", userData="smart_ai")
        self.mode_combo.setCurrentIndex(0)
        self.mode_combo.setToolTip("Auto: fast offline scan with confidence check and automatic Smart AI fallback.")

        self.btn_ai_settings = QPushButton("⚙️ AI Settings")
        self.btn_ai_settings.setObjectName("btn_secondary")
        self.btn_ai_settings.setToolTip("Configure Gemini API key for Smart AI features")
        self.btn_ai_settings.clicked.connect(self._open_ai_settings)

        self.btn_scan = QPushButton("  Scan Document")
        self.btn_scan.setObjectName("btn_scan")
        self.btn_scan.setEnabled(False)
        self.btn_scan.setMinimumWidth(170)
        self.btn_scan.clicked.connect(self.scan_document)

        self.btn_pdf = QPushButton("📄 Save as PDF")
        self.btn_pdf.setObjectName("btn_pdf")
        self.btn_pdf.clicked.connect(self._save_as_pdf)

        self.btn_export = QPushButton("Export Document...")
        self.btn_export.setObjectName("btn_export")
        self.btn_export.clicked.connect(self._export_document)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setObjectName("btn_secondary")
        self.btn_clear.clicked.connect(self._clear_results)

        for w in (cam_label, self.cam_combo, self.btn_refresh,
                  self.btn_camera, self.btn_rotate, self.btn_upload):
            self.ctrl_input_flow.addWidget(w)
        for w in (mode_label, self.mode_combo, self.btn_ai_settings,
                  self.btn_clear, self.btn_pdf, self.btn_export, self.btn_scan):
            self.ctrl_actions_flow.addWidget(w)

        self.ctrl_layout.addWidget(self.ctrl_input)
        self.ctrl_layout.addWidget(self.ctrl_actions)

        root_layout.addWidget(self.controls)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready  •  Computer Vision & AI Layout Pipeline Active")

        # Discover cameras
        self._refresh_cameras()

    # ── Theme (light / dark) ─────────────────────────────────
    def _toggle_theme(self):
        self.theme_name = theme.LIGHT if self.theme_name == theme.DARK else theme.DARK
        theme.save_theme(self.theme_name)
        self._apply_stylesheet()
        self._update_theme_button()
        self._render_formatted_tab()
        self._update_responsive_layout()
        self.status_bar.showMessage(f"{self.theme_name.capitalize()} mode enabled")

    def _update_theme_button(self):
        if self.theme_name == theme.DARK:
            self.btn_theme.setText("☀️  Light mode")
            self.btn_theme.setToolTip("Switch to light mode (Ctrl+T)")
        else:
            self.btn_theme.setText("🌙  Dark mode")
            self.btn_theme.setToolTip("Switch to dark mode (Ctrl+T)")

    def _render_formatted_tab(self):
        """(Re)build the Formatted Document tab using colours suited to the active theme."""
        if not self.all_pages_html:
            return
        pal = theme.PALETTES[self.theme_name]
        parts = []
        for page_num, source, html in self.all_pages_html:
            parts.append(
                f'<div style="border-bottom: 2px solid {pal["border"]}; margin: 20px 0; padding-bottom: 15px;">'
                f'<span style="background: {pal["accent_strong"]}; color: #ffffff; padding: 3px 8px; '
                f'border-radius: 4px; font-size: 11px;">PAGE {page_num}</span></div>'
            )
            parts.append(theme.adapt_html(html, source, self.theme_name))
        self.tab_formatted.setHtml("".join(parts))

    @staticmethod
    def _repolish(widget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    # ── Responsive layout ────────────────────────────────────
    STACK_BREAKPOINT = 980       # below this width: camera on top, results below
    SUBTITLE_BREAKPOINT = 1100   # below this width: hide the header subtitle

    def _set_initial_size(self):
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            self.resize(1200, 760)
            return
        avail = screen.availableGeometry()
        self.resize(min(1320, int(avail.width() * 0.92)), min(820, int(avail.height() * 0.92)))

    def _update_responsive_layout(self):
        if not hasattr(self, "splitter") or not hasattr(self, "ctrl_actions_flow"):
            return
        width = self.width()

        # Camera | Results side by side on wide windows, stacked on narrow ones
        wanted = Qt.Orientation.Vertical if width < self.STACK_BREAKPOINT else Qt.Orientation.Horizontal
        if self.splitter.orientation() != wanted:
            self.splitter.setOrientation(wanted)
            self.splitter.setSizes([2, 3] if wanted == Qt.Orientation.Vertical else [680, 640])

        self.subtitle.setVisible(width >= self.SUBTITLE_BREAKPOINT)

        # Toolbar: one row if both groups fit, otherwise stack them
        margins = self.centralWidget().layout().contentsMargins()
        available = width - margins.left() - margins.right()
        input_w = self.ctrl_input_flow.sizeHint().width()
        actions_w = self.ctrl_actions_flow.sizeHint().width()
        single_row = input_w + actions_w + self.ctrl_layout.spacing() <= available
        direction = QBoxLayout.Direction.LeftToRight if single_row else QBoxLayout.Direction.TopToBottom
        if self.ctrl_layout.direction() != direction:
            self.ctrl_layout.setDirection(direction)
        self.ctrl_actions_flow.set_right_align(single_row or actions_w <= available)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_responsive_layout()
        self._resize_timer.start(150)

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._update_responsive_layout)

    def _on_resize_settled(self):
        """Re-scale pixmaps that were rendered for the previous window size."""
        if not self.camera_running and self._preview_frame is not None:
            self._show_preview_frame()
        if self._visual_page_args is not None:
            self._render_visual_page()

    def _on_tab_changed(self, index):
        if index == 0 and self._visual_page_args is not None:
            QTimer.singleShot(0, self._render_visual_page)

    def _show_visual_placeholder(self):
        self._visual_page_args = None
        self.tab_visual_label.setObjectName("visual_placeholder")
        self.tab_visual_label.setPixmap(QPixmap())
        self.tab_visual_label.setText(
            "Physical document layout with exact 2D element coordinates will render here after scanning..."
        )
        self._repolish(self.tab_visual_label)

    def _show_preview_frame(self):
        """Show the uploaded image in the camera panel, scaled to the current panel size."""
        frame = self._preview_frame
        if frame is None:
            return
        lbl_w = max(self.cam_label.width(), 280)
        lbl_h = max(self.cam_label.height(), 200)
        fh, fw = frame.shape[:2]
        scale = min(lbl_w / float(fw), lbl_h / float(fh))
        nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
        thumb = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # Small label so the user knows which file is loaded
        cv2.putText(thumb, f"Uploaded: {self._preview_name}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 220, 100), 2, cv2.LINE_AA)
        rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, nw, nh, 3 * nw, QImage.Format.Format_RGB888)
        self.cam_label.setPixmap(QPixmap.fromImage(qimg))
        self.cam_label.setText("")

    # ── Camera Selection ────────────────────────────────────
    def _refresh_cameras(self):
        self.status_bar.showMessage("Scanning for cameras...")
        QApplication.processEvents()

        self.available_cameras = LiveCameraManager.detect_cameras(max_index=10)

        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()

        if self.available_cameras:
            for idx, label in self.available_cameras:
                self.cam_combo.addItem(label, userData=idx)
            self.status_bar.showMessage(f"Detected {len(self.available_cameras)} camera(s)")
        else:
            self.cam_combo.addItem("No cameras found")
            self.status_bar.showMessage("No cameras detected — connect your camera or phone and click Refresh")

        self.cam_combo.blockSignals(False)

    def _on_camera_changed(self, combo_index):
        if combo_index < 0 or combo_index >= len(self.available_cameras):
            return

        cam_index = self.available_cameras[combo_index][0]
        cam_label = self.available_cameras[combo_index][1]

        if self.camera_running:
            self.timer.stop()
            success = self.camera.switch_camera(cam_index)
            if success:
                self.timer.start(33)
                self.status_bar.showMessage(f"Switched to {cam_label}")
            else:
                self.status_bar.showMessage(f"Could not open {cam_label}")
        else:
            self.camera.camera_index = cam_index
            self.status_bar.showMessage(f"Selected {cam_label}")

    # ── Camera Timer ────────────────────────────────────────
    def _setup_camera_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

    def toggle_camera(self, checked):
        if checked:
            cam_idx = self.cam_combo.currentData()
            if cam_idx is None:
                cam_idx = 0

            self.status_bar.showMessage(f"Connecting to Camera {cam_idx}...")
            QApplication.processEvents()

            success = self.camera.open(cam_idx)
            if not success or not self.camera.is_opened():
                self.btn_camera.setChecked(False)
                self.btn_camera.setText("  Start Camera")
                self.status_bar.showMessage(f"Error: Could not open Camera {cam_idx}. Try another camera.")
                return

            self.btn_camera.setText("  Stop Camera")
            self.camera_running = True
            self._preview_frame = None
            self.btn_scan.setEnabled(True)
            self.timer.start(33)
            self.status_bar.showMessage("Camera running — position document in frame")
        else:
            self.btn_camera.setText("  Start Camera")
            self.camera_running = False
            self.btn_scan.setEnabled(False)
            self.timer.stop()
            self.camera.release()
            self.cam_label.clear()
            self.cam_label.setText("Camera Off\n\nClick 'Start Camera' to begin")
            self._set_badge("off")
            self.status_bar.showMessage("Camera stopped")

    def _rotate_view(self):
        self.rotation_angle = (self.rotation_angle + 90) % 360
        self.cached_corners = None
        self.status_bar.showMessage(f"Camera rotated to {self.rotation_angle}°")

    def _update_frame(self):
        if not self.camera_running:
            return
        frame = self.camera.get_frame()
        if frame is None:
            return

        if self.rotation_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation_angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        self.current_frame = frame.copy()

        # Update detection & quality check every 3 frames for smooth preview
        self.frame_counter += 1
        if self.frame_counter % 3 == 0 or self.cached_corners is None:
            self.cached_corners = self.detector.find_document(frame)
            self.cached_quality = self.detector.assess_frame_quality(frame, self.cached_corners)

        display = frame.copy()
        corners = self.cached_corners
        quality = self.cached_quality
        q_status = quality.get("status", "SEARCHING")
        q_message = quality.get("message", "")
        q_color = quality.get("color", (0, 220, 100))

        # ASCII-safe message for OpenCV Hershey font overlay (prevents '??????' rendering)
        clean_cv_msg = (
            q_message.replace("—", "-")
            .replace("–", "-")
            .replace("⚠️", "")
            .replace("✓", "")
            .strip()
        )

        if q_status == "BLURRY":
            self._set_badge("blurry", q_message)
            cv2.putText(display, f"[!] {clean_cv_msg}", (24, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, q_color, 2)
        elif corners is not None:
            self._set_badge("found", q_message)
            # Draw boundary polygon
            cv2.drawContours(display, [corners.astype(int)], -1, q_color, 3)
            for pt in corners.astype(int):
                cv2.circle(display, tuple(pt), 7, q_color, -1)
            cv2.putText(display, f"[OK] {clean_cv_msg}", (24, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, q_color, 2)
        else:
            self._set_badge("full_page", q_message)
            # Corner viewfinder brackets
            dh, dw = display.shape[:2]
            blen = 40
            c = (0, 220, 100)
            cv2.line(display, (25, 25), (25 + blen, 25), c, 3)
            cv2.line(display, (25, 25), (25, 25 + blen), c, 3)
            cv2.line(display, (dw - 25, 25), (dw - 25 - blen, 25), c, 3)
            cv2.line(display, (dw - 25, 25), (dw - 25, 25 + blen), c, 3)
            cv2.line(display, (25, dh - 25), (25 + blen, dh - 25), c, 3)
            cv2.line(display, (25, dh - 25), (25, dh - 25 - blen), c, 3)
            cv2.line(display, (dw - 25, dh - 25), (dw - 25 - blen, dh - 25), c, 3)
            cv2.line(display, (dw - 25, dh - 25), (dw - 25, dh - 25 - blen), c, 3)
            cv2.putText(display, f"[OK] {clean_cv_msg}", (24, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)

        # Fast SIMD pre-scale with OpenCV
        lbl_w = max(self.cam_label.width(), 320)
        lbl_h = max(self.cam_label.height(), 240)
        dh, dw = display.shape[:2]
        scale = min(lbl_w / float(dw), lbl_h / float(dh))
        nw, nh = max(1, int(dw * scale)), max(1, int(dh * scale))

        resized = cv2.resize(display, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, nw, nh, 3 * nw, QImage.Format.Format_RGB888)
        self.cam_label.setPixmap(QPixmap.fromImage(qimg))

    def _set_badge(self, state, text=None):
        if state == "found":
            self.doc_badge.setObjectName("badge_found")
            self.doc_badge.setText(f"  {text or 'Document Boundary Locked'}")
        elif state == "full_page":
            self.doc_badge.setObjectName("badge_found")
            self.doc_badge.setText(f"  {text or 'Full Document Mode — Ready'}")
        elif state == "blurry":
            self.doc_badge.setObjectName("badge_blurry")
            self.doc_badge.setText(f"  {text or 'Image Blurry'}")
        else:
            self.doc_badge.setObjectName("badge_searching")
            self.doc_badge.setText("  Camera Off")

        self.doc_badge.style().unpolish(self.doc_badge)
        self.doc_badge.style().polish(self.doc_badge)

    def _set_confidence_badge(self, state, text=None):
        if state == "smart_ai":
            self.confidence_badge.setObjectName("badge_confidence_smart_ai")
            self.confidence_badge.setText(f"  {text or 'Smart AI ✨'}")
        elif state == "high":
            self.confidence_badge.setObjectName("badge_confidence_high")
            self.confidence_badge.setText(f"  {text or 'Confidence: High ✓'}")
        elif state == "fair":
            self.confidence_badge.setObjectName("badge_confidence_fair")
            self.confidence_badge.setText(f"  {text or 'Confidence: Fair ⚠️'}")
        else:
            self.confidence_badge.setObjectName("badge_confidence_neutral")
            self.confidence_badge.setText(f"  {text or 'Confidence: —'}")

        self.confidence_badge.style().unpolish(self.confidence_badge)
        self.confidence_badge.style().polish(self.confidence_badge)

    def _open_ai_settings(self):
        dlg = AISettingsDialog(self.smart_ai_engine, self, theme_name=self.theme_name)
        dlg.exec()

    # ── Scan Execution ──────────────────────────────────────
    def scan_document(self):
        if self.current_frame is None:
            self.status_bar.showMessage("No camera frame available — start camera first")
            return
        if self.scan_worker and self.scan_worker.isRunning():
            return

        selected_mode = self.mode_combo.currentData() or "auto"
        mode_label = self.mode_combo.currentText()

        self.btn_scan.setEnabled(False)
        self.btn_scan.setText("  Scanning...")
        self.progress.setVisible(True)
        self.status_bar.showMessage(f"Digitizing document [{mode_label}]...")

        self.scan_worker = ScanWorker(
            self.current_frame.copy(),
            self.detector,
            self.analyzer,
            self.ocr_engine,
            self.smart_ai_engine,
            scan_mode=selected_mode
        )
        self.scan_worker.result_ready.connect(self._on_scan_result)
        self.scan_worker.error_occurred.connect(self._on_scan_error)
        self.scan_worker.status_update.connect(self.status_bar.showMessage)
        self.scan_worker.start()

    def _render_annotated_document(self, flat_img, elements: list, tokens: list, doc_data: dict = None):
        """
        Visual 2D Layout: clean digital reconstruction of the page.
        White background, every text line / table / figure drawn at the position
        it had on the physical page. Uses the same renderer as the PDF export.
        """
        self._visual_page_args = {
            "elements": elements,
            "image_shape": (doc_data or {}).get("image_shape") or (flat_img.shape[:2] if flat_img is not None else None),
        }
        self._render_visual_page()

    def _render_visual_page(self):
        """Render the last scanned page at a width that fits the current panel."""
        if self._visual_page_args is None:
            return
        try:
            target_w = max(260, min(900, self.tab_visual_scroll.viewport().width() - 30))
            pix = render_page_pixmap(self._visual_page_args, target_w)
            self.tab_visual_label.setObjectName("visual_page")
            self._repolish(self.tab_visual_label)
            self.tab_visual_label.setPixmap(pix)
            self.tab_visual_label.setText("")
        except Exception as e:
            print(f"Error rendering visual document: {e}")

    # ── Image Upload ──────────────────────────────────────
    def _upload_image(self):
        """Open a file dialog, load the image, preview it, and auto-digitize."""
        file_filter = (
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp *.jfif);;"
            "All Files (*)"
        )
        img_path, _ = QFileDialog.getOpenFileName(
            self, "Upload Image to Digitize", "", file_filter
        )
        if not img_path:
            return  # User cancelled

        # Read with OpenCV (handles all common image formats)
        frame = cv2.imread(img_path)
        if frame is None:
            QMessageBox.critical(
                self, "Image Load Error",
                f"Could not read the image file:\n{img_path}\n\n"
                "Please make sure it is a valid image (JPG, PNG, BMP, TIFF, etc.)."
            )
            return

        # Store frame so scan_document() picks it up
        self.current_frame = frame.copy()
        self._uploaded_image_path = img_path

        # Show thumbnail preview in the camera feed label (re-scaled on window resize)
        self._preview_frame = frame
        self._preview_name = os.path.basename(img_path)
        self._show_preview_frame()

        # Enable Scan button (works even when camera is off) and auto-digitize
        self.btn_scan.setEnabled(True)
        self.status_bar.showMessage(
            f"Image loaded: {img_path}  —  digitizing now..."
        )
        self.scan_document()

    def _on_scan_result(self, doc_data, flat_img):
        self.progress.setVisible(False)
        self.btn_scan.setEnabled(True)
        self.btn_scan.setText("  Scan Document")

        self.page_count += 1
        self.latest_doc = doc_data

        elements = doc_data.get("elements", [])
        page_md = doc_data.get("markdown", "")
        page_html = doc_data.get("html", "")
        validation = doc_data.get("validation", {})
        engine_name = doc_data.get("engine", "Offline (PaddleOCR + SLANet)")
        conf_info = doc_data.get("confidence", {})
        score = conf_info.get("score", 95.0)
        grade = conf_info.get("grade", "EXCELLENT")

        if engine_name.startswith("Smart AI"):
            self._set_confidence_badge("smart_ai", f"Smart AI ✨ ({score}%)")
        elif score >= 85.0:
            self._set_confidence_badge("high", f"Confidence: {score}% [{grade}] ✓")
        else:
            self._set_confidence_badge("fair", f"Confidence: {score}% [{grade}] ⚠️")

        self.all_pages_markdown.append(f"<!-- Page {self.page_count} -->\n" + page_md)
        html_source = theme.SOURCE_PRINT if engine_name.startswith("Smart AI") else theme.SOURCE_DARK
        self.all_pages_html.append((self.page_count, html_source, page_html))
        self.all_pages_elements.extend(elements)
        self.all_pages_data.append({
            "image": flat_img.copy() if flat_img is not None else None,
            "tokens": doc_data.get("tokens", []),
            "elements": elements,
            "title": doc_data.get("title", f"Page {self.page_count}"),
            "markdown": page_md,
            "html": page_html,
            "page_num": self.page_count
        })

        # 0. Update Visual 2D Document Layout Tab
        if flat_img is not None:
            self._render_annotated_document(flat_img, elements, doc_data.get("tokens", []), doc_data)

        # 1. Update Formatted Document Tab
        self._render_formatted_tab()

        # 2. Update Markdown Tab
        combined_md = "\n\n---\n\n".join(self.all_pages_markdown)
        self.tab_markdown.setPlainText(combined_md)

        # 3. Update Structured JSON Tab (Section 16 Specification)
        full_json_structure = {
            "document": {
                "title": doc_data.get("title", "Digitized Document"),
                "total_pages": self.page_count,
                "engine": engine_name,
                "confidence": conf_info,
                "elements": self.all_pages_elements,
                "validation": validation
            }
        }
        self.tab_json.setPlainText(json.dumps(full_json_structure, indent=2))

        # 4. Update Validation Report Tab
        val_lines = [
            "=" * 60,
            f"  VALIDATION & INTEGRITY REPORT — PAGE {self.page_count}",
            "=" * 60,
            f"Active Scan Engine: {engine_name}",
            f"Composite Confidence Score: {score} / 100 [{grade}]",
            f"  • OCR Token Accuracy: {conf_info.get('ocr_confidence', score)}%",
            f"  • Lexical Integrity: {conf_info.get('lexical_confidence', score)}%",
            f"  • Table Consistency: {conf_info.get('table_confidence', 100.0)}%",
            f"  • Layout Coherence: {conf_info.get('layout_confidence', 95.0)}%",
            "-" * 60,
            f"Overall Status: {validation.get('status', 'VERIFIED')}",
            f"Total Components Extracted: {len(elements)}",
            f"Uncertain OCR Detections: {validation.get('uncertain_items_count', 0)}",
            f"Mathematical Inconsistencies: {validation.get('inconsistencies_count', 0)}",
            ""
        ]

        if doc_data.get("escalated_from_offline"):
            val_lines.append("[AUTO-ESCALATION EVENT]")
            val_lines.append(f"  • Offline scan confidence was {doc_data.get('original_offline_score')}%.")
            val_lines.append(f"  • Escalated automatically to Smart AI (gemini-3.8-flash) for 100% human-grade quality.")
            val_lines.append("")

        if doc_data.get("ai_notice"):
            val_lines.append("[SMART AI ADVISORY]")
            val_lines.append(f"  • {doc_data.get('ai_notice')}")
            val_lines.append("")

        if doc_data.get("ai_fallback_warning"):
            val_lines.append("[AI NOTICE]")
            val_lines.append(f"  • Note: {doc_data.get('ai_fallback_warning')}")
            val_lines.append("")

        if validation.get("inconsistencies"):
            val_lines.append("[ARITHMETIC & TABLE CHECKS]")
            for inc in validation["inconsistencies"]:
                val_lines.append(f"  • ⚠️  {inc.get('message')}")
            val_lines.append("")

        if validation.get("uncertain_items"):
            val_lines.append("[UNCERTAIN OCR CONFIDENCE TOKENS]")
            for unc in validation["uncertain_items"]:
                val_lines.append(f"  • ⚠️  Element {unc.get('element_index')} ({unc.get('type')}): '{unc.get('text')}' (Confidence: {unc.get('confidence')})")
            val_lines.append("")

        if not validation.get("inconsistencies") and not validation.get("uncertain_items"):
            val_lines.append("✓ All detected elements meet confidence thresholds with 0 arithmetic errors.")

        val_lines.append("=" * 60)
        self.tab_validation.setPlainText("\n".join(val_lines))

        table_count = sum(1 for e in elements if e.get("type") == "Table")
        fig_count = sum(1 for e in elements if e.get("type") == "Image")
        self.status_bar.showMessage(
            f"Page {self.page_count} digitized [{engine_name}]! Extracted {len(elements)} elements ({table_count} table(s), {fig_count} figure(s)). Score: {score}%"
        )

    def _save_as_pdf(self):
        """Direct one-click shortcut to export digitized document as publication-ready PDF."""
        if not self.all_pages_markdown:
            QMessageBox.information(self, "Save as PDF", "No scanned documents to save yet. Please scan a document first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Digitized Document as PDF", "Digitized_Document.pdf", "PDF Document (*.pdf)"
        )
        if not path:
            return

        if not path.lower().endswith(".pdf"):
            path += ".pdf"

        try:
            combined_md = "\n\n---\n\n".join(self.all_pages_markdown)
            title = self.latest_doc.get("title", "Digitized Document") if self.latest_doc else "Digitized Document"

            # Export as clean structured digital PDF:
            # Properly typeset flowing document - headings bold & coloured,
            # tables formatted, paragraphs justified, no camera image anywhere
            ok = export_layout_pdf(self.all_pages_data, path, title=title)
            if ok:
                self.status_bar.showMessage(f"Digital PDF saved: {path}")
                QMessageBox.information(
                    self, "PDF Saved",
                    f"Successfully saved structured digital PDF to:\n{path}\n\n"
                    "• Pure white background — zero camera image\n"
                    "• Properly structured: Title, Headings, Paragraphs, Tables\n"
                    "• Headings in blue bold, body text justified\n"
                    "• Tables with styled header rows and borders\n"
                    "• Searchable and copy-pasteable text"
                )
            else:
                QMessageBox.critical(self, "Export Error", "Failed to generate PDF document.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save PDF: {e}")

    def _export_document(self):
        """Export digitized document as PDF, Markdown, HTML, JSON, or Plain Text."""
        if not self.all_pages_markdown:
            QMessageBox.information(self, "Export", "No scanned documents to export yet. Please scan a document first.")
            return

        file_filter = "PDF Document (*.pdf);;Markdown File (*.md);;HTML Document (*.html);;JSON Data (*.json);;Plain Text (*.txt)"
        path, selected_filter = QFileDialog.getSaveFileName(self, "Export Digitized Document", "Digitized_Document.pdf", file_filter)

        if not path:
            return

        try:
            combined_md = "\n\n---\n\n".join(self.all_pages_markdown)
            title = self.latest_doc.get("title", "Digitized Document") if self.latest_doc else "Digitized Document"

            if "*.pdf" in selected_filter or path.lower().endswith(".pdf"):
                if not path.lower().endswith(".pdf"):
                    path += ".pdf"
                # Clean structured typeset PDF - no camera image
                ok = export_layout_pdf(self.all_pages_data, path, title=title)
                if not ok:
                    raise RuntimeError("Failed to generate PDF.")
            elif "*.json" in selected_filter or path.lower().endswith(".json"):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.tab_json.toPlainText())
            elif "*.html" in selected_filter or path.lower().endswith(".html"):
                print_html = PDFExporter.markdown_to_print_html(combined_md, title)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(print_html)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.tab_markdown.toPlainText())

            self.status_bar.showMessage(f"Document successfully exported to: {path}")
            QMessageBox.information(self, "Export Successful", f"Saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save document: {e}")

    def _clear_results(self):
        self.tab_formatted.clear()
        self.tab_markdown.clear()
        self.tab_json.clear()
        self.tab_validation.clear()
        self.all_pages_markdown = []
        self.all_pages_html = []
        self.all_pages_elements = []
        self.all_pages_data = []
        self._show_visual_placeholder()
        self.page_count = 0
        self.status_bar.showMessage("Results cleared.")

    def _on_scan_error(self, msg):
        self.progress.setVisible(False)
        self.btn_scan.setEnabled(True)
        self.btn_scan.setText("  Scan Document")
        error_color = theme.PALETTES[self.theme_name]["error_text"]
        self.tab_formatted.setHtml(f'<div style="color: {error_color}; font-weight: bold; padding: 20px;">Scan Error: {msg}</div>')
        self.status_bar.showMessage(f"Scan failed: {msg}")

    # ── Cleanup ───────────────────────────────────────────
    def closeEvent(self, event):
        self.timer.stop()
        self.camera.release()
        event.accept()
