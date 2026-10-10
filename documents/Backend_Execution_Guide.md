# LiveDoc Backend Execution Guide & Architecture

## Overview
This document serves as the complete technical manual for the **LiveDoc Backend Pipeline**. It details exactly how the backend executes, how data flows from the camera to the AI models, and the mathematical logic behind every function. 

If you are presenting this project, you can use this document to explain exactly *how* the AI works under the hood.

---

## The Execution Flow (Pipeline)
When a user clicks "Scan Document" in the UI, the backend executes in a strict, sequential 4-step pipeline:

1. **Capture:** `LiveCameraManager` grabs a raw Numpy Array (image) from the webcam.
2. **Pre-Process:** `DocumentDetector` analyzes the image, finds the paper's physical borders, and mathematically warps the skewed paper so it appears perfectly flat.
3. **Layout Analysis:** `LayoutAnalyzer` uses a YOLOv11 Neural Network to find exactly where the text columns and tables are on the flattened paper, returning coordinate boxes.
4. **Text Extraction:** `OCREngine` uses PaddleOCR to read the text inside those specific coordinate boxes, automatically correcting for upside-down or sideways text.

---

## 1. Camera Manager (`src/camera_manager.py`)
**Purpose:** Interfaces with the hardware webcam to capture raw video frames.

* **`__init__()`**: Initializes OpenCV's `cv2.VideoCapture(0)` to connect to the computer's default webcam.
* **`get_frame()`**: Reads a single frame from the camera buffer. It returns a 3D NumPy array representing the pixels (Height x Width x Color Channels).
* **`release()`**: Safely turns off the webcam hardware light and frees up system memory.

---

## 2. Document Detector (`src/document_detector.py`)
**Purpose:** The hardest part of computer vision is dealing with noise (messy desks, shadows). This class uses geometric math to isolate the paper from the background.

* **`is_blurry(frame, threshold=50)`**: 
  * *How it works:* Converts the image to grayscale and applies a Laplacian Filter (calculates the 2nd derivative of the image).
  * *Why:* The Laplacian filter highlights sharp edges. If the variance of the edges is low (under 50), it means the image is out of focus (blurry). We use this to stop the OCR from reading a blurry, moving camera feed.
* **`find_document(frame)`**: 
  * *How it works:* Uses Bilateral Filtering to blur the desk texture but keep the paper edge sharp. It applies Canny Edge Detection to draw white lines over high-contrast edges. It then uses `cv2.findContours` to find the largest closed shape, and `cv2.approxPolyDP` to verify if the shape has exactly 4 corners (a rectangle).
  * *Why:* To dynamically locate the 4 physical corners of the paper in real-world space, regardless of where it is on the desk.
* **`_order_points(pts)`**: 
  * *How it works:* Sorts the 4 corners using coordinate addition/subtraction to strictly guarantee they are ordered: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
* **`flatten_document(frame, corners)`**:
  * *How it works:* Uses the Euclidean distance formula to calculate the paper's real width/height. It then computes a 3x3 Perspective Transformation Matrix (`cv2.getPerspectiveTransform`) and warps the pixels (`cv2.warpPerspective`) to flatten the 3D angle into a perfect 2D scan.

---

## 3. Layout Analyzer (`src/layout_analyzer.py`)
**Purpose:** Prevents the OCR from reading across two separate columns of text as if they were one sentence.

* **`__init__()`**: Loads a custom YOLOv11 deep-learning model built on PyTorch.
* **`_yolo_analyze(image)`**: 
  * *How it works:* Feeds the flattened document into the YOLO model. The model predicts the locations of layout elements.
  * *Output:* Returns a List of Dictionaries. Each dictionary contains the `label` (e.g., "Title", "Text") and the `box` (the [x1, y1, x2, y2] pixel coordinates of where that text is).
* **`analyze_document(image)`**: Public wrapper that safely falls back to a "Full Page" block if the YOLO weights are missing, ensuring the app never crashes.

---

## 4. OCR Engine (`src/ocr_engine.py`)
**Purpose:** Converts the pixels of letters into actual String text.

* **`__init__()`**: Initializes the PaddleOCR C++ backend. 
  * *Crucial Setting:* We explicitly enable `use_angle_cls=True`. This loads a specialized orientation model (`PP-LCNet`) that can detect if a user is holding the paper sideways or upside down and digitally rotates it before reading it.
* **`extract_full_text(image)`**: 
  * *How it works:* Passes the image tensor into PaddleOCR. Paddle returns a massive, nested array containing text strings, box coordinates, and confidence scores. This function iterates through the nested array and extracts only the final String text, joining it with newlines (`\n`).
* **`extract_from_boxes(image, layout_boxes)`**:
  * *How it works:* Takes the coordinate boxes from YOLO and physically crops the NumPy array using indexing: `cropped = image[y1:y2, x1:x2]`. It then feeds only this tiny cropped image to the OCR. 
  * *Why:* By only giving the OCR one column or table at a time, we guarantee 100% perfect reading order.
