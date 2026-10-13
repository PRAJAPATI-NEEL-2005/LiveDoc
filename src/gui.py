import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QTextEdit, QHBoxLayout, QVBoxLayout, QSplitter, QStatusBar,
    QProgressBar, QFrame, QSizePolicy, QComboBox
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QIcon

# Import backend modules
from src.camera_manager import LiveCameraManager
from src.document_detector import DocumentDetector
from src.layout_analyzer import LayoutAnalyzer
from src.ocr_engine import OCREngine


# ─────────────────────────────────────────────────────────
# Worker thread: runs OCR in background so UI never freezes
# ─────────────────────────────────────────────────────────
class ScanWorker(QThread):
    result_ready = pyqtSignal(list, object)   # (data list, flattened image)
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
            self.status_update.emit("Checking blur...")
            if self.detector.is_blurry(self.frame):
                self.error_occurred.emit("Image is too blurry. Hold the camera steady.")
                return

            self.status_update.emit("Finding document edges...")
            corners = self.detector.find_document(self.frame)
            if corners is None:
                self.error_occurred.emit("No document detected. Make sure the paper is fully visible on a contrasting surface.")
                return

            self.status_update.emit("Applying perspective correction...")
            flat = self.detector.flatten_document(self.frame, corners)

            self.status_update.emit("Running layout analysis...")
            boxes = self.analyzer.analyze(flat)

            self.status_update.emit("Extracting text with OCR...")
            data = self.ocr_engine.extract_from_boxes(flat, boxes)

            self.result_ready.emit(data, flat)
        except Exception as e:
            self.error_occurred.emit(f"Pipeline error: {str(e)}")


# ─────────────────────────────────────────────────────────
# Main Application Window
# ─────────────────────────────────────────────────────────
class LiveDocApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiveDoc — Real-Time Intelligent Document Digitizer")
        self.setMinimumSize(1280, 780)

        # Backend objects — don't open any camera yet; user picks from dropdown
        self.camera = LiveCameraManager(0)
        self.available_cameras = []  # [(index, label), ...]
        self.detector = DocumentDetector()
        self.analyzer = LayoutAnalyzer()
        self.ocr_engine = OCREngine()

        # State
        self.camera_running = False
        self.current_frame = None
        self.scan_worker = None

        self._apply_stylesheet()
        self._build_ui()
        self._setup_camera_timer()

    # ── Stylesheet ────────────────────────────────────────
    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0f1117;
                color: #e2e8f0;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }
            QLabel#title_label {
                font-size: 22px;
                font-weight: bold;
                color: #7dd3fc;
                padding: 4px 0px;
            }
            QLabel#subtitle_label {
                font-size: 12px;
                color: #64748b;
                padding-bottom: 6px;
            }
            QLabel#cam_feed {
                background-color: #1e2330;
                border: 2px solid #2d3748;
                border-radius: 12px;
            }
            QLabel#section_header {
                font-size: 12px;
                font-weight: bold;
                color: #94a3b8;
                letter-spacing: 1px;
                padding: 6px 0px 2px 0px;
            }
            QTextEdit {
                background-color: #1a1f2e;
                border: 2px solid #2d3748;
                border-radius: 10px;
                color: #e2e8f0;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
                padding: 12px;
                selection-background-color: #3b82f6;
            }
            QPushButton {
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
                border: none;
            }
            QPushButton#btn_camera {
                background-color: #1e40af;
                color: #ffffff;
            }
            QPushButton#btn_camera:hover {
                background-color: #2563eb;
            }
            QPushButton#btn_camera:checked {
                background-color: #991b1b;
            }
            QPushButton#btn_camera:checked:hover {
                background-color: #dc2626;
            }
            QPushButton#btn_scan {
                background-color: #065f46;
                color: #ffffff;
                font-size: 15px;
                padding: 14px 28px;
            }
            QPushButton#btn_scan:hover {
                background-color: #059669;
            }
            QPushButton#btn_scan:disabled {
                background-color: #1e2330;
                color: #4b5563;
            }
            QPushButton#btn_clear {
                background-color: #1e2330;
                color: #94a3b8;
                border: 1px solid #374151;
            }
            QPushButton#btn_clear:hover {
                background-color: #374151;
                color: #e2e8f0;
            }
            QStatusBar {
                background-color: #0a0d14;
                color: #64748b;
                font-size: 12px;
                padding: 2px 8px;
                border-top: 1px solid #1e2330;
            }
            QProgressBar {
                background-color: #1e2330;
                border-radius: 4px;
                border: none;
                height: 4px;
                text-align: center;
                color: transparent;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 4px;
            }
            QFrame#divider {
                background-color: #2d3748;
            }
            QLabel#status_dot {
                font-size: 11px;
                padding: 4px 10px;
                border-radius: 10px;
                background-color: #1e2330;
                color: #64748b;
            }
            QLabel#badge_blurry {
                background-color: #7f1d1d;
                color: #fca5a5;
                padding: 3px 10px;
                border-radius: 8px;
                font-size: 11px;
                font-weight: bold;
            }
            QLabel#badge_found {
                background-color: #064e3b;
                color: #6ee7b7;
                padding: 3px 10px;
                border-radius: 8px;
                font-size: 11px;
                font-weight: bold;
            }
            QComboBox {
                background-color: #1a1f2e;
                color: #e2e8f0;
                border: 2px solid #2d3748;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 13px;
                font-weight: bold;
                min-width: 200px;
            }
            QComboBox:hover {
                border-color: #3b82f6;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #1a1f2e;
                color: #e2e8f0;
                border: 2px solid #3b82f6;
                selection-background-color: #1e40af;
                padding: 4px;
            }
            QPushButton#btn_refresh {
                background-color: #1e2330;
                color: #94a3b8;
                border: 1px solid #374151;
                padding: 8px 14px;
                font-size: 12px;
            }
            QPushButton#btn_refresh:hover {
                background-color: #374151;
                color: #e2e8f0;
            }
            QLabel#badge_searching {
                background-color: #1e2330;
                color: #64748b;
                padding: 3px 10px;
                border-radius: 8px;
                font-size: 11px;
            }
        """)

    # ── UI Layout ─────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(20, 16, 20, 12)
        root_layout.setSpacing(12)

        # Header
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("LiveDoc")
        title.setObjectName("title_label")
        sub = QLabel("Real-Time Intelligent Document Digitizer  •  CV + YOLO + PaddleOCR")
        sub.setObjectName("subtitle_label")
        sub.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self.doc_badge = QLabel("  Searching for document...")
        self.doc_badge.setObjectName("badge_searching")

        header_layout.addWidget(title)
        header_layout.addSpacing(12)
        header_layout.addWidget(sub)
        header_layout.addStretch()
        header_layout.addWidget(self.doc_badge)
        root_layout.addWidget(header)

        # Progress bar (hidden by default)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # Indeterminate
        self.progress.setVisible(False)
        root_layout.addWidget(self.progress)

        # Main content: Camera | Results
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("QSplitter::handle { background-color: #2d3748; }")

        # ── Left: Camera panel ──
        cam_panel = QWidget()
        cam_layout = QVBoxLayout(cam_panel)
        cam_layout.setContentsMargins(0, 0, 8, 0)
        cam_layout.setSpacing(8)

        cam_header = QLabel("LIVE CAMERA")
        cam_header.setObjectName("section_header")
        cam_layout.addWidget(cam_header)

        self.cam_label = QLabel()
        self.cam_label.setObjectName("cam_feed")
        self.cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cam_label.setText("Camera Off\n\nClick 'Start Camera' to begin")
        self.cam_label.setMinimumSize(600, 450)
        self.cam_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        cam_layout.addWidget(self.cam_label, 1)

        splitter.addWidget(cam_panel)

        # ── Right: Results panel ──
        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(8, 0, 0, 0)
        result_layout.setSpacing(8)

        res_header = QLabel("EXTRACTED DOCUMENT")
        res_header.setObjectName("section_header")
        result_layout.addWidget(res_header)

        self.result_text = QTextEdit()
        self.result_text.setPlaceholderText(
            "Extracted text, tables, and structured data will appear here after scanning...\n\n"
            "Tip: Position the document flat on a contrasting surface for best results."
        )
        self.result_text.setReadOnly(False)
        result_layout.addWidget(self.result_text, 1)

        splitter.addWidget(result_panel)
        splitter.setSizes([700, 500])
        root_layout.addWidget(splitter, 1)

        # ── Bottom controls ──
        controls = QWidget()
        ctrl_layout = QHBoxLayout(controls)
        ctrl_layout.setContentsMargins(0, 4, 0, 0)
        ctrl_layout.setSpacing(12)

        self.btn_camera = QPushButton("  Start Camera")
        self.btn_camera.setObjectName("btn_camera")
        self.btn_camera.setCheckable(True)
        self.btn_camera.setMinimumWidth(160)
        self.btn_camera.clicked.connect(self.toggle_camera)

        self.btn_scan = QPushButton("  Scan Document")
        self.btn_scan.setObjectName("btn_scan")
        self.btn_scan.setEnabled(False)
        self.btn_scan.setMinimumWidth(200)
        self.btn_scan.clicked.connect(self.scan_document)

        self.btn_clear = QPushButton("Clear Results")
        self.btn_clear.setObjectName("btn_clear")
        self.btn_clear.clicked.connect(self.result_text.clear)

        ctrl_layout.addWidget(self.btn_camera)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.btn_clear)
        ctrl_layout.addWidget(self.btn_scan)
        root_layout.addWidget(controls)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready  •  Python 3.10  •  PaddleOCR + YOLO backend loaded")

    # ── Camera Timer ──────────────────────────────────────
    def _setup_camera_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

    def toggle_camera(self, checked):
        if checked:
            self.btn_camera.setText("  Stop Camera")
            self.camera_running = True
            self.btn_scan.setEnabled(True)
            self.timer.start(30)  # 30ms = ~33 FPS
            self.status_bar.showMessage("Camera started — show a document to the camera")
        else:
            self.btn_camera.setText("  Start Camera")
            self.camera_running = False
            self.btn_scan.setEnabled(False)
            self.timer.stop()
            self.cam_label.clear()
            self.cam_label.setText("Camera Off\n\nClick 'Start Camera' to begin")
            self._set_badge("off")
            self.status_bar.showMessage("Camera stopped")

    def _update_frame(self):
        frame = self.camera.get_frame()
        if frame is None:
            return
        self.current_frame = frame.copy()

        # Draw document detection overlay in real time
        display = frame.copy()
        is_blur = self.detector.is_blurry(frame)
        corners = self.detector.find_document(frame)

        if is_blur:
            self._set_badge("blurry")
            cv2.putText(display, "BLURRY — Hold Steady", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 255), 2)
        elif corners is not None:
            self._set_badge("found")
            # Draw green quadrilateral outline
            cv2.drawContours(display, [corners.astype(int)], -1, (0, 220, 100), 3)
            # Draw corner dots
            for pt in corners.astype(int):
                cv2.circle(display, tuple(pt), 8, (0, 220, 100), -1)
            cv2.putText(display, "Document Detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 100), 2)
        else:
            self._set_badge("searching")
            cv2.putText(display, "Searching for document...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (160, 160, 160), 2)

        # Convert BGR → RGB → QPixmap and display
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Scale to fit label while keeping aspect ratio
        pixmap = pixmap.scaled(
            self.cam_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.cam_label.setPixmap(pixmap)

    def _set_badge(self, state):
        if state == "found":
            self.doc_badge.setObjectName("badge_found")
            self.doc_badge.setText("  Document Detected")
        elif state == "blurry":
            self.doc_badge.setObjectName("badge_blurry")
            self.doc_badge.setText("  Image Blurry")
        elif state == "searching":
            self.doc_badge.setObjectName("badge_searching")
            self.doc_badge.setText("  Searching for document...")
        else:
            self.doc_badge.setObjectName("badge_searching")
            self.doc_badge.setText("  Camera Off")
        # Force style refresh
        self.doc_badge.style().unpolish(self.doc_badge)
        self.doc_badge.style().polish(self.doc_badge)

    # ── Scan ─────────────────────────────────────────────
    def scan_document(self):
        if self.current_frame is None:
            self.status_bar.showMessage("No frame available — start the camera first")
            return
        if self.scan_worker and self.scan_worker.isRunning():
            return  # Already scanning

        self.btn_scan.setEnabled(False)
        self.btn_scan.setText("  Scanning...")
        self.progress.setVisible(True)
        self.result_text.setPlaceholderText("Scanning... please wait")
        self.status_bar.showMessage("Scanning document...")

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

    def _on_scan_result(self, data, flat_img):
        self.progress.setVisible(False)
        self.btn_scan.setEnabled(True)
        self.btn_scan.setText("  Scan Document")

        # Format results into readable output
        lines = []
        lines.append("=" * 50)
        lines.append("  DIGITIZED DOCUMENT")
        lines.append("=" * 50)
        lines.append("")

        for i, block in enumerate(data):
            btype = block.get("type", "Unknown").upper()
            text = block.get("text", "").strip()
            if text:
                lines.append(f"[{btype}]")
                lines.append(text)
                lines.append("")

        if not any(b.get("text", "").strip() for b in data):
            lines.append("No text could be extracted.")
            lines.append("Try: better lighting, hold document flat, avoid glare.")

        lines.append("=" * 50)
        self.result_text.setPlainText("\n".join(lines))
        self.status_bar.showMessage(
            f"Scan complete — {len(data)} region(s) extracted  |  "
            f"Regions: {', '.join(b['type'] for b in data)}"
        )

    def _on_scan_error(self, msg):
        self.progress.setVisible(False)
        self.btn_scan.setEnabled(True)
        self.btn_scan.setText("  Scan Document")
        self.result_text.setPlainText(f"[ERROR]\n\n{msg}")
        self.status_bar.showMessage(f"Scan failed: {msg}")

    # ── Cleanup ───────────────────────────────────────────
    def closeEvent(self, event):
        self.timer.stop()
        self.camera.release()
        event.accept()
