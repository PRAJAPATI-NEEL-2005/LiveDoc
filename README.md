# LiveDoc: Real-Time Intelligent Document Digitizer

**LiveDoc** is a real-time, domain-agnostic document digitization and structural understanding platform powered by Computer Vision, Deep Learning Layout Analysis, PaddleOCR, and Rule-Based Integrity Validation.

Unlike traditional flat scanners or mobile camera apps that merely save static images or PDFs, **LiveDoc actively deconstructs physical documents into native digital components** (Titles, Headings, Paragraphs, Lists, and Reconstructed Markdown/HTML Data Tables) while preserving authentic vertical reading flow.

---

## Key Capabilities & Architecture

LiveDoc operates via a modular 5-stage pipeline:

```
┌─────────────────────────┐
│  Live Camera Feed       │  (Webcam / Iriun 1080p, Threaded Ring Buffer)
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│ Document Detection &    │  (Adaptive Edge Filtering, 4-Point Contour Lock,
│ Perspective Correction  │   Warp Transform, Multi-Metric Blur/Glare Check)
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│ Layout Decomposition    │  (Morphological Grid Detection & Bounding Boxes:
│                         │   Tables, Section Blocks, Header Banners)
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│ Intelligent OCR &       │  (PaddleOCR text detection & recognition,
│ Table Reconstruction    │   Outline & Heading Classification, Matrix Tables)
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│ Validation & Integrity  │  (Confidence Auditing, Arithmetic Discrepancy Checks,
│ Multi-Page Export       │   Interactive PyQt6 GUI with Markdown / HTML export)
└─────────────────────────┘
```

### 1. Real-Time Camera & Positioning Guidance (`src/camera_manager.py`, `src/document_detector.py`)
- Background capture thread eliminating UI freezes.
- Auto-detects 1080p/720p hardware resolutions (supports physical webcams and phone cameras via Iriun/DroidCam).
- Real-time quality evaluation:
  - **Sharpness/Blur check** via downscaled Laplacian variance ($> 20.0$).
  - **Glare assessment** monitoring saturated pixel ratios ($> 8\%$).
  - **Centering and proximity advice** alerting users if the document is too far or off-center.
- 4-point convex hull detection with perspective transform to flatten skewed captures into crisp top-down scans.

### 2. High-Precision Layout Decomposition (`src/layout_analyzer.py`)
- Morphological horizontal and vertical projection kernels to isolate table borders, horizontal rules, and distinct content sections.
- Graceful dual-engine architecture: uses custom YOLO models (`models/yolo11_doc.pt`) when weights are present, seamlessly falling back to high-precision OpenCV morphological analysis.

### 3. Domain-Agnostic OCR & Structural Reading Flow (`src/ocr_engine.py`, `src/table_extractor.py`)
- **Zero hardcoding / Domain-Agnostic**: Contains no subject-specific vocabulary, dictionary overrides, or hardcoded strings. Handles mathematical texts, invoices, academic literature, and technical manuals purely through geometric and typographic principles.
- **Outline & Heading Detection**:
  - Distinguishes running page headers (`line_y < 0.14 * H` with broad horizontal span).
  - Identifies hierarchical multi-level numbered sections (`1.5`, `2.1`).
  - Identifies Roman numeral subsections (`i)`, `ii)`, `iv)`) and letter bullets (`A.`, `B.`).
  - Distinguishes true Title-Cased headings from body numbered list items (`1.`, `2.`, `3.`) and sentence fragments ending in prepositions.
  - Excludes mathematical formulas and equations from heading promotion.
- **Optical Noise & Thin-Paper Bleed-Through Suppression**:
  - Automatically drops stray optical artifacts with low confidence ($< 0.60$).
  - Prunes isolated single-token bleed-through text from 50–60 GSM paper under live camera lighting.
  - Requires high OCR confidence ($\ge 0.78$) for section heading classification.
- **In-Line Table Matrix Reconstruction**:
  - Crops table regions with targeted $1.5\times$ resolution scaling for 100% cell character recovery (including single-character values like `T` and `F`).
  - Automatically flushes preceding paragraphs when encountering a table vertically, preserving table placement directly beneath introducing sentences.
  - Synthesizes formatted Markdown tables and styled HTML tables.

### 4. Validation & Integrity Engine (`src/validator.py`)
- **Principle**: Extract $\rightarrow$ Validate $\rightarrow$ Flag $\rightarrow$ Preserve Original $\rightarrow$ Report.
- Never silently mutates or guesses uncertain characters.
- Flags tokens with OCR confidence $< 0.72$ for user inspection in the dedicated **Validation Report** tab.
- Runs automated row-level arithmetic checks ($Qty \times Unit\ Price == Total$) and grand total column sums on tabular financial documents.

### 5. Multi-Page Workflow & Desktop GUI (`src/gui.py`)
- Modern dark-themed PyQt6 interface.
- Live camera preview with real-time green boundary overlay and diagnostic status indicators.
- Multi-Tab Result Viewer:
  - **Formatted Document**: Rendered HTML with styled headings, clean paragraphs, and borders.
  - **Markdown / Plain Text**: Clean GitHub-Flavored Markdown.
  - **Structured JSON**: Machine-readable token coordinates, bounding boxes, element types, and confidences.
  - **Validation Report**: Summary of overall document status (`VERIFIED` vs `NEEDS_VERIFICATION`), confidence warnings, and consistency checks.
- Multi-Page Document Session:
  - "Scan Page", "Add Next Page", "Clear / New Doc".
  - One-click export of complete multi-page documents to `.md` or `.html`.

---

## Directory Structure

```
LiveDoc/
├── main.py                     # GUI Entrypoint
├── test_pipeline.py            # Comprehensive 22-point validation test suite
├── train_layout_model.py       # Training pipeline for DocLayNet / YOLO11 doc models
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation
├── src/
│   ├── __init__.py
│   ├── camera_manager.py       # Threaded multi-resolution camera ingestion
│   ├── document_detector.py    # Edge detection, quality checks & perspective warp
│   ├── layout_analyzer.py      # Bounding box layout decomposition
│   ├── ocr_engine.py           # PaddleOCR extraction, outline hierarchy & noise filters
│   ├── table_extractor.py      # Matrix cell clustering & table reconstruction
│   ├── validator.py            # Confidence auditing & arithmetic validation
│   └── gui.py                  # PyQt6 graphical interface
└── models/                     # Custom trained YOLO weights (gitignored)
```

---

## Installation & Setup

### Prerequisites
- macOS, Linux, or Windows
- Python 3.10 or 3.11 (Conda recommended)
- Webcam, laptop camera, or mobile camera via Iriun Webcam

### 1. Environment Setup
```bash
# Clone the repository
git clone -b Viraj <repo-url>
cd LiveDoc

# Create and activate conda environment
conda create -n livedoc python=3.10 -y
conda activate livedoc

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation
Run the complete 22-point integration test suite:
```bash
python test_pipeline.py
```
Expected output:
```
============================================================
  SUMMARY
============================================================
  22/22 checks passed
  ✅ OpenCV
  ✅ NumPy
  ✅ PyQt6
  ✅ Ultralytics (YOLO)
  ✅ PaddleOCR
  ...
============================================================
```

### 3. Launch LiveDoc
```bash
python main.py
```

---

## Usage Guide

1. **Select Camera**: Use the camera dropdown in the top-left toolbar to choose your webcam or connected phone camera (e.g. `Camera 0`, `Iriun Webcam`).
2. **Align Document**: Hold or place your document within view. The live feed will display a green boundary overlay once a clean rectangular page contour is locked.
3. **Check Status**: Ensure the status pill reads **"Document Boundary Locked — Ready to Scan"**.
4. **Scan Page**: Click **"📸 Scan Current Frame"**. The system will flatten perspective, analyze layout, perform OCR, reconstruct tables, and populate the tabs.
5. **Multi-Page Scanning**: Click **"➕ Add Next Page"** to append additional pages into a unified document.
6. **Export**: Click **"💾 Export Markdown"** or **"💾 Export HTML"** to save the final structured digitized document.
