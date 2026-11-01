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
import json
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QTextEdit, QTextBrowser, QHBoxLayout, QVBoxLayout, QSplitter,
    QStatusBar, QProgressBar, QFrame, QSizePolicy, QComboBox,
    QTabWidget, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QIcon

# Import backend modules
from src.camera_manager import LiveCameraManager
from src.document_detector import DocumentDetector
from src.layout_analyzer import LayoutAnalyzer
from src.ocr_engine import OCREngine
from src.pdf_exporter import PDFExporter


# ─────────────────────────────────────────────────────────
# Worker thread: runs OCR in background so UI never freezes
# ─────────────────────────────────────────────────────────
class ScanWorker(QThread):
    result_ready = pyqtSignal(dict, object)   # (structured_doc_dict, flat_image)
    error_occurred = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self, frame, detector, analyzer, ocr_engine):
        super().__init__()
        self.frame = frame
        self.detector = detector
        self.analyzer = analyzer
        self.ocr_engine = ocr_engine

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

            self.status_update.emit("Analyzing document layout (Tables, Titles, Text)...")
            boxes = self.analyzer.analyze(flat)

            self.status_update.emit("Extracting structured text & reconstructing tables...")
            doc_data = self.ocr_engine.extract_structured_document(flat, boxes)

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
        self.setMinimumSize(1320, 820)

        # Backend objects
        self.camera = LiveCameraManager(0)
        self.available_cameras = []
        self.detector = DocumentDetector()
        self.analyzer = LayoutAnalyzer()
        self.ocr_engine = OCREngine()

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
        self.all_pages_html = []
        self.all_pages_elements = []

        self._apply_stylesheet()
        self._build_ui()
        self._setup_camera_timer()

    # ── Stylesheet ────────────────────────────────────────
    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0d1117;
                color: #e2e8f0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }
            QLabel#title_label {
                font-size: 22px;
                font-weight: 800;
                color: #38bdf8;
                padding: 4px 0px;
            }
            QLabel#subtitle_label {
                font-size: 12px;
                color: #94a3b8;
                padding-bottom: 6px;
            }
            QLabel#cam_feed {
                background-color: #161b22;
                border: 2px solid #30363d;
                border-radius: 12px;
            }
            QLabel#section_header {
                font-size: 12px;
                font-weight: bold;
                color: #94a3b8;
                letter-spacing: 1px;
                padding: 4px 0px;
            }
            QTabWidget::pane {
                border: 2px solid #30363d;
                border-radius: 8px;
                background-color: #161b22;
            }
            QTabBar::tab {
                background: #0d1117;
                color: #94a3b8;
                font-weight: bold;
                padding: 8px 16px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background: #161b22;
                color: #38bdf8;
                border-bottom: 2px solid #38bdf8;
            }
            QTabBar::tab:hover {
                color: #e2e8f0;
            }
            QTextEdit, QTextBrowser {
                background-color: #161b22;
                border: none;
                color: #e2e8f0;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
                padding: 14px;
                selection-background-color: #1e40af;
            }
            QPushButton {
                border-radius: 8px;
                padding: 10px 18px;
                font-size: 13px;
                font-weight: 600;
                border: none;
            }
            QPushButton#btn_camera {
                background-color: #1d4ed8;
                color: #ffffff;
            }
            QPushButton#btn_camera:hover {
                background-color: #2563eb;
            }
            QPushButton#btn_camera:checked {
                background-color: #b91c1c;
            }
            QPushButton#btn_camera:checked:hover {
                background-color: #dc2626;
            }
            QPushButton#btn_scan {
                background-color: #059669;
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                padding: 12px 24px;
            }
            QPushButton#btn_scan:hover {
                background-color: #10b981;
            }
            QPushButton#btn_scan:disabled {
                background-color: #1f2937;
                color: #4b5563;
            }
            QPushButton#btn_export {
                background-color: #0284c7;
                color: #ffffff;
            }
            QPushButton#btn_export:hover {
                background-color: #0369a1;
            }
            QPushButton#btn_pdf {
                background-color: #dc2626;
                color: #ffffff;
                font-weight: bold;
            }
            QPushButton#btn_pdf:hover {
                background-color: #b91c1c;
            }
            QPushButton#btn_secondary {
                background-color: #21262d;
                color: #c9d1d9;
                border: 1px solid #30363d;
            }
            QPushButton#btn_secondary:hover {
                background-color: #30363d;
                color: #ffffff;
            }
            QStatusBar {
                background-color: #0a0d14;
                color: #8b949e;
                font-size: 12px;
                padding: 3px 10px;
                border-top: 1px solid #21262d;
            }
            QProgressBar {
                background-color: #21262d;
                border-radius: 4px;
                border: none;
                height: 4px;
            }
            QProgressBar::chunk {
                background-color: #38bdf8;
                border-radius: 4px;
            }
            QComboBox {
                background-color: #161b22;
                color: #e2e8f0;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 13px;
                min-width: 200px;
            }
            QComboBox:hover {
                border-color: #38bdf8;
            }
            QLabel#badge_searching {
                background-color: #21262d;
                color: #8b949e;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: bold;
            }
            QLabel#badge_found {
                background-color: #064e3b;
                color: #6ee7b7;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: bold;
            }
            QLabel#badge_blurry {
                background-color: #7f1d1d;
                color: #fca5a5;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: bold;
            }
        """)

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
        sub = QLabel("Real-Time Intelligent Document Digitizer  •  Computer Vision + YOLO + PaddleOCR")
        sub.setObjectName("subtitle_label")
        sub.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self.doc_badge = QLabel("  Searching for document...")
        self.doc_badge.setObjectName("badge_searching")

        header_layout.addWidget(title)
        header_layout.addSpacing(14)
        header_layout.addWidget(sub)
        header_layout.addStretch()
        header_layout.addWidget(self.doc_badge)
        root_layout.addWidget(header)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        root_layout.addWidget(self.progress)

        # Main splitter (Camera | Results Tabs)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("QSplitter::handle { background-color: #30363d; }")

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
        self.cam_label.setMinimumSize(600, 460)
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

        result_layout.addWidget(self.tabs, 1)
        splitter.addWidget(result_panel)
        splitter.setSizes([680, 640])
        root_layout.addWidget(splitter, 1)

        # ── Bottom Controls ──
        controls = QWidget()
        ctrl_layout = QHBoxLayout(controls)
        ctrl_layout.setContentsMargins(0, 4, 0, 0)
        ctrl_layout.setSpacing(10)

        # Camera selection
        cam_label = QLabel("Camera:")
        cam_label.setStyleSheet("color: #94a3b8; font-weight: bold;")
        self.cam_combo = QComboBox()
        self.cam_combo.currentIndexChanged.connect(self._on_camera_changed)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setObjectName("btn_secondary")
        self.btn_refresh.clicked.connect(self._refresh_cameras)

        self.btn_camera = QPushButton("  Start Camera")
        self.btn_camera.setObjectName("btn_camera")
        self.btn_camera.setCheckable(True)
        self.btn_camera.setMinimumWidth(150)
        self.btn_camera.clicked.connect(self.toggle_camera)

        self.btn_rotate = QPushButton("Rotate 90°")
        self.btn_rotate.setObjectName("btn_secondary")
        self.btn_rotate.clicked.connect(self._rotate_view)

        self.btn_scan = QPushButton("  Scan Document")
        self.btn_scan.setObjectName("btn_scan")
        self.btn_scan.setEnabled(False)
        self.btn_scan.setMinimumWidth(180)
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

        ctrl_layout.addWidget(cam_label)
        ctrl_layout.addWidget(self.cam_combo)
        ctrl_layout.addWidget(self.btn_refresh)
        ctrl_layout.addSpacing(6)
        ctrl_layout.addWidget(self.btn_camera)
        ctrl_layout.addWidget(self.btn_rotate)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.btn_clear)
        ctrl_layout.addWidget(self.btn_pdf)
        ctrl_layout.addWidget(self.btn_export)
        ctrl_layout.addWidget(self.btn_scan)

        root_layout.addWidget(controls)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready  •  Computer Vision & AI Layout Pipeline Active")

        # Discover cameras
        self._refresh_cameras()

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

        was_running = self.camera_running
        if was_running:
            self.timer.stop()

        success = self.camera.switch_camera(cam_index)
        if success:
            self.status_bar.showMessage(f"Switched to {cam_label}")
        else:
            self.status_bar.showMessage(f"Could not open {cam_label}")

        if was_running and success:
            self.timer.start(40)

    # ── Camera Timer ────────────────────────────────────────
    def _setup_camera_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

    def toggle_camera(self, checked):
        if checked:
            self.btn_camera.setText("  Stop Camera")
            self.camera_running = True
            self.btn_scan.setEnabled(True)
            self.timer.start(40)
            self.status_bar.showMessage("Camera running — position document in frame")
        else:
            self.btn_camera.setText("  Start Camera")
            self.camera_running = False
            self.btn_scan.setEnabled(False)
            self.timer.stop()
            self.cam_label.clear()
            self.cam_label.setText("Camera Off\n\nClick 'Start Camera' to begin")
            self._set_badge("off")
            self.status_bar.showMessage("Camera stopped")

    def _rotate_view(self):
        self.rotation_angle = (self.rotation_angle + 90) % 360
        self.cached_corners = None
        self.status_bar.showMessage(f"Camera rotated to {self.rotation_angle}°")

    def _update_frame(self):
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

        if q_status == "BLURRY":
            self._set_badge("blurry", q_message)
            cv2.putText(display, f"⚠️ {q_message}", (24, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, q_color, 2)
        elif corners is not None:
            self._set_badge("found", q_message)
            # Draw boundary polygon
            cv2.drawContours(display, [corners.astype(int)], -1, q_color, 3)
            for pt in corners.astype(int):
                cv2.circle(display, tuple(pt), 7, q_color, -1)
            cv2.putText(display, f"✓ {q_message}", (24, 45),
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
            cv2.putText(display, f"✓ {q_message}", (24, 45),
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

    # ── Scan Execution ──────────────────────────────────────
    def scan_document(self):
        if self.current_frame is None:
            self.status_bar.showMessage("No camera frame available — start camera first")
            return
        if self.scan_worker and self.scan_worker.isRunning():
            return

        self.btn_scan.setEnabled(False)
        self.btn_scan.setText("  Scanning...")
        self.progress.setVisible(True)
        self.status_bar.showMessage("Digitizing document with layout understanding...")

        self.scan_worker = ScanWorker(
            self.current_frame.copy(),
            self.detector,
            self.analyzer,
            self.ocr_engine
        )
        self.scan_worker.result_ready.connect(self._on_scan_result)
        self.scan_worker.error_occurred.connect(self._on_scan_error)
        self.scan_worker.status_update.connect(self.status_bar.showMessage)
        self.scan_worker.start()

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

        self.all_pages_markdown.append(f"<!-- Page {self.page_count} -->\n" + page_md)
        self.all_pages_html.append(f'<div style="border-bottom: 2px solid #30363d; margin: 20px 0; padding-bottom: 15px;"><span style="background: #1e40af; color: #fff; padding: 3px 8px; border-radius: 4px; font-size: 11px;">PAGE {self.page_count}</span></div>' + page_html)
        self.all_pages_elements.extend(elements)

        # 1. Update Formatted Document Tab
        combined_html = "".join(self.all_pages_html)
        self.tab_formatted.setHtml(combined_html)

        # 2. Update Markdown Tab
        combined_md = "\n\n---\n\n".join(self.all_pages_markdown)
        self.tab_markdown.setPlainText(combined_md)

        # 3. Update Structured JSON Tab (Section 16 Specification)
        full_json_structure = {
            "document": {
                "title": doc_data.get("title", "Digitized Document"),
                "total_pages": self.page_count,
                "elements": self.all_pages_elements,
                "validation": validation
            }
        }
        self.tab_json.setPlainText(json.dumps(full_json_structure, indent=2))

        # 4. Update Validation Report Tab
        val_lines = [
            "=" * 55,
            f"  VALIDATION & INTEGRITY REPORT — PAGE {self.page_count}",
            "=" * 55,
            f"Overall Status: {validation.get('status', 'VERIFIED')}",
            f"Total Components Extracted: {len(elements)}",
            f"Uncertain OCR Detections: {validation.get('uncertain_items_count', 0)}",
            f"Mathematical Inconsistencies: {validation.get('inconsistencies_count', 0)}",
            ""
        ]

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

        val_lines.append("=" * 55)
        self.tab_validation.setPlainText("\n".join(val_lines))

        table_count = sum(1 for e in elements if e.get("type") == "Table")
        self.status_bar.showMessage(
            f"Page {self.page_count} digitized! Extracted {len(elements)} elements ({table_count} table(s))."
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
            ok = PDFExporter.markdown_to_pdf(combined_md, path, doc_title=title)
            if ok:
                self.status_bar.showMessage(f"Document successfully saved as PDF: {path}")
                QMessageBox.information(self, "PDF Saved", f"Successfully saved PDF to:\n{path}")
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
                ok = PDFExporter.markdown_to_pdf(combined_md, path, doc_title=title)
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
        self.page_count = 0
        self.status_bar.showMessage("Results cleared.")

    def _on_scan_error(self, msg):
        self.progress.setVisible(False)
        self.btn_scan.setEnabled(True)
        self.btn_scan.setText("  Scan Document")
        self.tab_formatted.setHtml(f'<div style="color: #ef4444; font-weight: bold; padding: 20px;">Scan Error: {msg}</div>')
        self.status_bar.showMessage(f"Scan failed: {msg}")

    # ── Cleanup ───────────────────────────────────────────
    def closeEvent(self, event):
        self.timer.stop()
        self.camera.release()
        event.accept()
