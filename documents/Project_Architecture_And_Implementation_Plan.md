# LiveDoc: Real-Time Intelligent Document Digitizer
## Comprehensive Project Architecture, Deep Research, and Implementation Plan

This document outlines the state-of-the-art approach to building the **LiveDoc** Real-Time Intelligent Document Digitizer. Based on deep research into current Document AI and Computer Vision frameworks (as of 2026), this guide details the pipeline, architecture, accuracy testing, and constraints provided by the instructor.

---

## 1. Core Architecture & Pipeline (OOP Approach)

As per the constraints, the entire project will be designed using **Object-Oriented Programming (OOP)**. This ensures modularity, testability, and scalability. 

The pipeline consists of the following core classes:

### `LiveCameraManager`
- **Responsibility:** Capture real-time frames from the webcam.
- **Actions:** Stream frames, adjust resolution, manage camera buffers.

### `DocumentDetector` (Quality & Localization)
- **Responsibility:** Find the document in the live frame and assess if it is clear.
- **Actions:**
  - **Edge Detection:** Detect document boundaries (OpenCV contour detection).
  - **Blur Detection:** Calculate Laplacian variance to ensure the frame is sharp before processing.
  - **Perspective Transform:** Apply 4-point perspective warp to flatten the document.

### `LayoutAnalyzer` (Document Understanding)
- **Responsibility:** Analyze the flattened document to find text, tables, images, and charts.
- **Actions:** Uses a deep learning model to draw bounding boxes and classify regions.

### `OCREngine` (Text & Data Extraction)
- **Responsibility:** Extract text from the regions identified by the `LayoutAnalyzer`.
- **Actions:** Run OCR on text blocks and table cells. Calculate confidence scores.

### `DigitizedDocument` (Data Structure)
- **Responsibility:** Store the structured output (Text, Tables, Images).
- **Actions:** Export to JSON, PDF, or Markdown.

### `MetricsEvaluator` (Accuracy Testing)
- **Responsibility:** Compare the extracted data against ground-truth data to measure accuracy.
- **Actions:** Calculate CER, WER, and mAP (detailed in Section 4).

---

## 2. Research & Technology Comparisons: "Options vs. Best Choices"

We researched multiple approaches for each component of the pipeline. Here is the comparison and our recommended **Best Choice**.

### A. Document Layout Analysis (Finding Text vs Tables vs Images)
*   **Option 1: Heuristic / OpenCV contours.** (Finding blocks based on whitespace). Very fast, but terrible for complex layouts (multi-column, tables).
*   **Option 2: LayoutLMv3.** Excellent for scanned PDFs, but very heavy and slow for *real-time camera feeds*.
*   **Option 3: YOLOv11 / DocLayout-YOLO (Trained on DocLayNet).** A lightweight object detection model specifically trained to find document elements (Title, Text, Table, Figure).
*   **Our Recommendation:** **DocLayout-YOLO or YOLOv11.** *Why?* It is the current state-of-the-art for *real-time* execution, capable of running at 60+ FPS while accurately parsing complex layouts.

### B. Optical Character Recognition (OCR)
*   **Option 1: Tesseract OCR.** Extremely fast, lightweight, and doesn't require a GPU. However, it struggles with noisy images, handwriting, and complex fonts.
*   **Option 2: EasyOCR.** Good balance of accuracy and ease of use, built on PyTorch. Slower than Tesseract but much more accurate on natural images.
*   **Option 3: PaddleOCR.** State-of-the-art for complex layouts, table structure recognition, and multi-language support.
*   **Our Recommendation:** **PaddleOCR.** *Why?* While Tesseract is fast, LiveDoc requires "Intelligent" understanding. PaddleOCR not only reads text beautifully but has built-in table structure recognition, making it the most powerful tool for this pipeline. 

### C. GUI (Graphical User Interface) Constraint Analysis
Your instructor restricted you to using **Python for GUI** (No external web frameworks like React, Next.js, or HTML/JS). 
*   **Option 1: Tkinter.** Built into Python. Very easy to use, but looks very outdated (like Windows 95).
*   **Option 2: Kivy.** Good for mobile apps, but overkill and complex for desktop.
*   **Option 3: PyQt6 / PySide6.** Powerful, professional, and looks native.
*   **Option 4: CustomTkinter.** A modern wrapper around Tkinter that adds dark mode and rounded corners.
*   **Our Recommendation:** **CustomTkinter or PyQt6.** 
*   **⚠️ IMPORTANT NOTE TO INSTRUCTOR/TEAM:** While PyQt6/CustomTkinter fulfills the Python-only GUI constraint, creating the "World's Best" modern, dynamic AI application is heavily restricted by this. The current industry standard for AI applications is a Python Backend (FastAPI) paired with a Web Frontend (React/Streamlit) for dynamic document rendering and interactive bounding boxes. **If you want a truly world-class UI, we highly recommend requesting an exception to this constraint from your instructor to use Streamlit or a Web UI.** If the constraint is strict, we will proceed with **PyQt6** for the most professional native look.

---

## 3. Step-by-Step Implementation Guide

Follow these exact steps to build the project smoothly:

**Step 1: Setup & OOP Skeleton**
- Create the Python classes defined in Section 1. Do not add AI yet; just make sure the classes can pass data to each other.

**Step 2: Camera & Quality Check (OpenCV)**
- Implement `LiveCameraManager`. 
- Implement `DocumentDetector` to find the 4 corners of a piece of paper using `cv2.findContours`. 
- Add blur detection (`cv2.Laplacian(img, cv2.CV_64F).var()`). Only capture if the score is above a threshold.

**Step 3: Perspective Warp**
- Use `cv2.getPerspectiveTransform` and `cv2.warpPerspective` to flatten the detected paper into a perfect rectangle.

**Step 4: Layout Analysis Integration**
- Load a pre-trained YOLO model (like `yolov8n` fine-tuned on documents).
- Pass the flattened image to YOLO to get bounding boxes for Text, Tables, and Images.

**Step 5: OCR & Structuring**
- Pass the cropped bounding boxes to PaddleOCR.
- Store the results in the `DigitizedDocument` object.

**Step 6: GUI Integration**
- Wrap the entire pipeline in a PyQt6 or CustomTkinter window. Show the live camera feed on the left, and the structured digitized output on the right.

---

## 4. Testing Accuracy & Evaluation Metrics

To prove to your instructor that the project works accurately, we must test it using scientific metrics. Here is how we will evaluate the system:

### 1. Document Detection Accuracy (Intersection over Union - IoU)
- **What it measures:** How accurately did the camera find the edges of the paper?
- **How to test:** Compare the bounding box drawn by our system against a perfectly drawn manual bounding box. An IoU > 0.8 is excellent.

### 2. Layout Analysis Accuracy (mAP - Mean Average Precision)
- **What it measures:** Did the system correctly identify a Table as a Table, and Text as Text?
- **How to test:** Run the system on a test dataset (like DocLayNet). Calculate the mAP at 50% IoU. 

### 3. Text Extraction Accuracy (CER & WER)
- **CER (Character Error Rate):** Calculates how many characters were read incorrectly. (Formula: `(Substitutions + Deletions + Insertions) / Total Characters`). *Goal: < 5%*
- **WER (Word Error Rate):** Calculates how many whole words were read incorrectly. *Goal: < 10%*
- **How to test:** We will create a `MetricsEvaluator` class. We will pass an image of a known document, extract the text, and programmatically compare it to a `.txt` file containing the perfect, true text using the Levenshtein distance algorithm.

---
**Summary:** By following this OOP architecture, utilizing PaddleOCR and YOLO for state-of-the-art extraction, and strictly testing our CER/WER accuracy, LiveDoc will be a robust, highly intelligent system.
