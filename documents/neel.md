# Neel's Work Log: LiveDoc Frontend, UI/UX & Pipeline Integration

This document tracks all the work completed by **Neel Prajapati** on the **LiveDoc: Real-Time Intelligent Document Digitizer** project. It details the frontend architecture, desktop UI/UX engineering, layout and OCR integration, table matrix reconstruction, integrity validation, and multi-page workflow systems.

---

## 1. Project Role & Responsibilities

As defined in the project master blueprint and task division:
- **Primary Owner:** Desktop Graphical User Interface (`src/gui.py`, `main.py`).
- **Core Processing Pipeline:** Layout Analysis & Classification (`src/layout_analyzer.py`), YOLOv11 model training pipeline (`train_layout_model.py`).
- **Data Structuring & Verification:** Table Extraction & 2D Matrix Reconstructor (`src/table_extractor.py`), Document Integrity Validator (`src/validator.py`).
- **Pipeline Integration:** Connecting camera ingestion, perspective flattening, layout analysis, OCR engine, and presentation layers.
- **Multi-Page & Export Engine:** Multi-page document aggregation and export into Markdown, HTML, JSON, and Plain Text.

---

## 2. Desktop GUI Architecture (`src/gui.py`)

### Problem Statement
Running deep learning models (PaddleOCR and YOLO) on high-resolution image crops in real time causes severe graphical thread blocking (1–3 second GUI freezes). Traditional desktop apps become unresponsive during capture, resulting in poor user experience.

### Architectural Solution: Asynchronous Multi-Threaded Pipeline
Implemented `ScanWorker` as a dedicated background worker inheriting from `PyQt6.QtCore.QThread`:
- Decouples UI event loop (main thread) from compute-intensive OCR, edge detection, and layout analysis.
- Communicates back to the UI thread using Qt signals (`result_ready`, `status_update`, `error_occurred`).
- Continuous 30 FPS camera preview updates via non-blocking `QTimer` (`tick` every 30ms).
- Live camera viewport with dynamic frame rotation ($90^\circ$ steps) and real-time document contour viewfinder.

### Multi-Tab Structured Presentation Layer
Replaced primitive text boxes with a tabbed interface (`QTabWidget`):
1. **Formatted Document Tab (`QTextBrowser`):** Renders clean, styled HTML with CSS-formatted tables, colored headings, and visual tags.
2. **Markdown Text Tab (`QTextEdit`):** Provides raw GitHub-Flavored Markdown for easy copy-pasting into note-taking apps (Obsidian, Notion).
3. **Structured JSON Tab (`QTextEdit`):** Outputs standardized, machine-readable JSON containing page indices, element coordinates, bounding boxes, element types, and confidence scores.
4. **Validation Report Tab (`QTextEdit`):** Real-time summary of document confidence health, flagged low-confidence tokens, and arithmetic inconsistencies.

### Real-Time Quality Guidance Overlay
Embedded dynamic visual guidance directly above the live viewfinder:
- **Sharpness Monitoring:** Evaluates Laplacian variance in real time; warns if camera motion causes blur.
- **Glare Assessment:** Monitors overexposed pixel ratios; prompts user to adjust light angle.
- **Proximity & Centering:** Alerts if document occupies too little screen area or is skewed off-center.
- **Visual Feedback:** Color-coded status pills (Orange = Warning, Amber = Centering, Green = Locked & Ready).

---

## 3. Table Matrix Reconstruction (`src/table_extractor.py`)

### Problem Statement
Standard OCR reads text left-to-right, losing table row and column relationships. When numbers from adjacent cells merge, tabular documents (invoices, truth tables, balance sheets) become completely corrupted.

### The Algorithm: 2D Geometric Slot Clustering
1. **Row Clustering:** Tokens are sorted by $Y$-coordinate and grouped into physical rows based on dynamic vertical thresholds derived from median character heights.
2. **Column Slot Discovery:** The header row establishes reference $X$-coordinate column centers. For tables where headers are absent or merged, a 1D clustering pass groups token $X$-centers.
3. **Grid Projection:** Each token is mapped to the nearest column center Euclidean distance. Multi-word tokens within a cell are joined with single whitespace.
4. **Markdown & HTML Generation:** Produces clean Markdown pipes (`|`) with alignment markers (`:---:`) and responsive HTML tables with dark-mode styling.

---

## 4. Integrity Validation & Confidence Auditing (`src/validator.py`)

### The Guiding Principle
**Extract $\rightarrow$ Validate $\rightarrow$ Flag $\rightarrow$ Preserve Original $\rightarrow$ Report.**
The system must never silently alter or invent text. If a token is uncertain or an arithmetic equation doesn't add up, it must flag the discrepancy for human review while keeping the original OCR output intact.

### Validation Rules
- **OCR Confidence Auditing:** Tokens with recognition confidence $< 0.72$ are flagged and listed in the Validation Report.
- **Tabular Arithmetic Verification:** Automatically detects column headers (`Qty`, `Price`, `Total`) and verifies:
  $$\text{Quantity} \times \text{Unit Price} == \text{Printed Total}$$
- **Grand Total Summation:** Verifies that individual row line items sum to the printed Grand Total.

---

## 5. Multi-Page Scanning & Export Engine

- **Page Accumulator:** Users can click "Add Next Page" to continuously scan multiple pages of a multi-page report or book.
- **One-Click Export:** Digitized multi-page documents can be saved as:
  - `.md` (Markdown with embedded page separators)
  - `.html` (Complete standalone HTML document with embedded CSS)
  - `.json` (Full document hierarchy with bounding boxes)
  - `.txt` (Plain text extraction)

---

## 6. Verification & Test Suite

Created comprehensive automated test suites:
- `test_pipeline.py`: 22-point end-to-end integration test validating camera, detector, layout analysis, OCR, and end-to-end extraction.
- `tests/test_table_extractor.py`: Unit tests for row clustering, column slot alignment, and HTML/Markdown table rendering.
- `tests/test_validator.py`: Unit tests for confidence thresholding and arithmetic consistency checks.
