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

## 6. Digital Vector PDF Reconstruction & 1-to-1 Physical Positioning

### Problem Statement
Standard document scanner apps create "sandwich PDFs" that plaster the original grainy, shadowed, or skewed camera photo as a background raster and hide invisible OCR text beneath it. This fails accessibility, wastes print toner, looks unprofessional, and prevents clean digital editing.

### Architectural Solution (`src/pdf_exporter.py`, `src/layout_renderer.py`)
Engineered a pure digital vector document reconstruction pipeline:
- **Zero Photo Background**: Rebuilds the document from scratch on clean, vector-rendered white pages (`#FFFFFF`).
- **1-to-1 Physical Coordinate Mapping**: Computes physical coordinate scalings ($X_{\text{scale}} = \frac{W_{\text{target}}}{W_{\text{source}}}$, $Y_{\text{scale}} = \frac{H_{\text{target}}}{H_{\text{source}}}$) to position headings, paragraphs, and tables at their exact physical page locations without drift or overlapping.
- **Dynamic Text Reflow & Page Budgeting**: Implements automatic font scaling (8.5pt–10.5pt) and line wrapping within column boundaries. Tables and diagrams are bounded to prevent cross-page splits.

---

## 7. Visual 2D Layout Renderer & Desktop Tab (`src/layout_renderer.py`, `src/gui.py`)

- **Interactive 2D Canvas Tab**: Added the **Visual 2D Layout** tab in the desktop GUI, rendering an on-screen preview of the reconstructed digital page matching the exact physical layout.
- **Aspect-Ratio Preserved Image Rendering**: Automatically computes proportional scaling (`scale = min(avail_w / img_w, avail_h / img_h, 1.0)`), centers diagrams on the page, applies a subtle slate-gray border, and draws italicized captions below.
- **Spatial Collision Suppression**: Automatically suppresses paragraphs or stray tokens that spatially overlap with diagram or table bounding boxes, preventing internal chart axis labels from polluting document prose.

---

## 8. Multi-Format Image File Upload Pipeline (`src/gui.py`)

### Problem Statement
Users need to digitize existing document photos and scans stored on their disk, not just live webcam captures.

### Implementation
- Added a dedicated purple **📂 Upload Image** button (`btn_upload`) in the desktop GUI toolbar.
- **Multi-Format Ingestion**: Supports `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.webp`, and `.jfif`.
- **Live Viewfinder Preview & Badge**: Upon selection, the uploaded image is displayed in the main viewfinder with a prominent blue badge showing the filename, image dimensions, and "IMAGE UPLOAD MODE".
- **Instant Auto-Digitization**: Automatically processes the uploaded file through the entire pipeline (Quality Assessment $\rightarrow$ Perspective Flattening $\rightarrow$ Layout Decomposition $\rightarrow$ OCR / AI $\rightarrow$ PDF/Markdown Generation) and populates all 5 output tabs.

---

## 9. Markdown & LaTeX Typography Sanitization (`src/text_markup.py`)

- **Syntax Leakage Prevention**: Built `clean_display_text()` to scrub raw Markdown symbols (`**`, `*`, `__`, `_`, `~~`, `` ` ``, `##`, `>`) from physical document displays so raw code syntax never leaks into rendered PDFs or visual tabs.
- **Mathematical Entity Conversion**: Converts LaTeX and ASCII math notations into publication-grade Unicode symbols:
  - Arrows: `\xrightarrow`, `->` $\rightarrow$ `→`, `<-` $\rightarrow$ `←`, `=>` $\rightarrow$ `⇒`
  - Greek letters: `\delta` $\rightarrow$ `δ`, `\lambda` $\rightarrow$ `λ`, `\Sigma` $\rightarrow$ `Σ`
  - Logic operators: `&&` $\rightarrow$ `∧` (in logic contexts) or `&` (in words)
  - HTML entities: `&amp;` $\rightarrow$ `&`, `&nbsp;`, `&emsp;` $\rightarrow$ spaces.

---

## 10. Local Model Accuracy & Intelligent Diagram Detection (`src/ocr_engine.py`, `src/document_detector.py`)

### Problem Statement
In local (offline) mode, PaddleOCR missed small text, subscripts, superscripts, Greek letters, and transition arrows. Furthermore, technical diagrams (state machines, flowcharts, circuits) were completely omitted or fragmented into stray one-word paragraphs.

### Engineering Solutions Implemented:
1. **Adaptive CLAHE Illumination Equalization (`src/document_detector.py`)**:
   - Integrated Contrast-Limited Adaptive Histogram Equalization (`clipLimit=1.6`, tile size `8x8`) into `preprocess_for_ocr()`.
   - Equalizes phone-camera shadows and uneven illumination, boosting faint text and thin line strokes without introducing threshold noise.
2. **High-Resolution Fine-Detail OCR Parameters (`src/ocr_engine.py`)**:
   - `det_limit_side_len`: Raised from `1280` to `1600` to prevent downsampling on high-resolution camera images.
   - `det_db_box_thresh`: Tuned to `0.30` to detect small arrowheads and fine mathematical operators.
   - `det_db_unclip_ratio`: Increased to `1.7` to prevent clipping ascenders and descenders (`q`, `g`, `p`, $\delta$, $\lambda$).
3. **Intelligent Diagram & Graphic Segmentation Engine (`src/ocr_engine.py`)**:
   - **Selective Prose Masking**: Only masks long prose tokens (`len(txt) > 6`), preserving state circles, arrowheads, and short labels (`q₀`, `q₁`, `0/A`, `1/B`).
   - **Diagram Bridge Kernel**: Uses a `(17, 17)` morphological bridge kernel to unite connected components (nodes, transitions, flowchart boxes) into coherent diagram blocks.
   - **Contour Boundary Clearing & `cv2.RETR_LIST`**: Zeros out 12px outer image borders and uses `cv2.RETR_LIST` so outer page edges never trap or hide interior diagrams.
   - **Table vs Diagram Safeguard**: Strictly excludes clusters containing figure captions (`Fig.`, `Figure`, `Diagram`) from being classified as tables.
   - **Caption Anchoring**: Matches captions (`(?i)\b(?:fig(?:\.|ure)?|chart|diagram|graph|scheme)\b.*`) and links them directly to the graphic element.
   - **Diagram Token Suppression**: Automatically assigns and suppresses all internal diagram tokens (`0/A`, `1/B`, `q₀`, `Fig.2.69`) so they never leak as stray prose paragraphs.
4. **Automata & Mathematical Notation Healing (`src/ocr_engine.py`)**:
   - Automata states: `q0` / `qo` $\rightarrow$ `q₀`, `q1` $\rightarrow$ `q₁`, `q2` $\rightarrow$ `q₂`, `qn` $\rightarrow$ `qₙ`, `a0` $\rightarrow$ `a₀`, `b1` $\rightarrow$ `b₁`.
   - Greek letters & transitions: `8[` / `8(` $\rightarrow$ `δ(`, `1[` / `1(` $\rightarrow$ `λ(`, `->` $\rightarrow$ `→`.
   - Printed OCR typo corrections: `follwing` $\rightarrow$ `following`, `ino` $\rightarrow$ `into`, `equialent` $\rightarrow$ `equivalent`, `machin` $\rightarrow$ `machine`.
   - Roman numeral regex fix: Requires explicit punctuation (`i.`, `ii)`, `(a)`) to prevent bare words like `M on input a.` from being misclassified as headings.
5. **Layout Analyzer Hardening (`src/layout_analyzer.py`)**:
   - Added upper bound checks to prevent whole-page borders from being mistaken for tables.
   - Added safe fallback wrappers around `ultralytics YOLO` imports to guarantee zero crashes on Windows DLL loading issues.

---

## 11. Verification & Test Suite Status

- **Main Pipeline Integration Test (`test_pipeline.py`)**: **24/24 PASS**
- **Unit Test Suite (`tests/`)**: **12/12 PASS** (`test_sandwich_pdf`, `test_table_extractor`, `test_validator`)
- **Figure Test Suite (`run_figure_test.py`)**: **PASS** (`PDF ok: True`)
- **Real Scanned Image Pipeline Test (`run_image_pipeline_test.py`)**: **56/56 checks with 0 failures** across diverse real-world images (automata diagrams, dense comprehension paragraphs, book covers, two-page spreads)
- **User Benchmark Scan (`page2-81.jpeg`)**: State transition diagram cleanly extracted and cropped to `documents/figures/figure_1.png` with caption `Fig.2.69` and clean mathematical formula reconstruction.
