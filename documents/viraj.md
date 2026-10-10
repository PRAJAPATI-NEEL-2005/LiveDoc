# Viraj's Work Log: LiveDoc Project

This document tracks all the work completed by Viraj on the **LiveDoc: Real-Time Intelligent Document Digitizer** project. It serves as a record of methods used, problems solved, and files created so that Neel and other team members can easily understand the progress.

---

## [2026-10-10] - Initial Setup & Camera Integration

### 1. Project Initialization & Folder Structure
- **What was done:** Set up the foundational directory structure for the entire project.
- **Method:** Created `src/` for OOP classes, `models/` for YOLO weights, and `documents/` for documentation. Created empty template files for both Viraj and Neel (`camera_manager.py`, `document_detector.py`, `layout_analyzer.py`, `ocr_engine.py`, `gui.py`).
- **Why:** To ensure a clean separation of concerns and maintain a proper OOP architecture from day one.

### 2. Dependency Management & Environment Setup
- **What was done:** Created `requirements.txt` and successfully configured the Python environment.
- **Method:** 
  - Defined dependencies: `opencv-python`, `ultralytics`, `paddlepaddle`, `paddleocr`, `PyQt6`, `numpy`.
  - **Critical Fix:** Encountered an error where `paddlepaddle` and `ultralytics` failed to install because the default Python version (3.13) was too new. Resolved this by explicitly creating a Conda environment locked to **Python 3.10** (`conda create -n livedoc python=3.10`).
- **Why:** Resolving environment errors early ensures Neel can clone the repo and install dependencies without breaking his system.

### 3. Camera Integration (`src/camera_manager.py`)
- **What was done:** Implemented the `LiveCameraManager` class to handle real-time webcam feeds.
- **Method:** 
  - Used OpenCV (`cv2.VideoCapture(0)`) to access the system's primary webcam.
  - Implemented `get_frame()` to continuously read frames as NumPy arrays.
  - Added a quick-test block at the bottom of the script using `cv2.imshow` so the camera feed can be tested instantly by running `python src/camera_manager.py`.
- **Why:** This is the first critical step of the pipeline. Neel's AI models will rely on these NumPy array frames to perform layout analysis and OCR.

---

*Note for Neel: Before starting your AI work, make sure you read `documents/plan.md` and set up your Conda environment using Python 3.10 as outlined in Section 3!*
