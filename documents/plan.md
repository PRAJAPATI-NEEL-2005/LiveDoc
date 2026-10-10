# LiveDoc: Master Implementation & Collaboration Plan

This is the finalized, comprehensive blueprint for building **LiveDoc: Real-Time Intelligent Document Digitizer**. This document divides the project precisely between **Viraj** and **Neel**, defining the exact files, classes, functions, and inputs/outputs they need to build. By following this strictly, your independent code will connect flawlessly when merged.

---

## 1. Finalized Technology Stack

*   **Core Language:** Python 3.10+
*   **Image Processing & Camera Control:** OpenCV (`cv2`) 
*   **Document Layout Analysis (Deep Learning):** YOLOv11 / DocLayout-YOLO (trained on DocLayNet)
*   **Optical Character Recognition (OCR):** PaddleOCR
*   **Graphical User Interface (GUI):** PyQt6 (Primary) / CustomTkinter (Fallback)

---

## 2. Project Folder Structure

Set up this exact folder structure in the repository before writing code:

```text
LiveDoc/
│
├── main.py                   # Run by Viraj to start the app
├── requirements.txt          # Shared dependencies list
│
├── src/                      
│   ├── __init__.py
│   ├── camera_manager.py     # Viraj's Task
│   ├── document_detector.py  # Viraj's Task
│   ├── layout_analyzer.py    # Neel's Task
│   ├── ocr_engine.py         # Neel's Task
│   └── gui.py                # Viraj's Task
│
├── models/                   # Folder to store downloaded YOLO weights (.pt files)
└── documents/                # Folder for documentation
```

---

## 3. Strict Collaboration Plan: Who Does What?

To ensure the pipeline connects properly without bugs, every function must have an agreed-upon **Input** and **Output**. 

### Part A: VIRAJ'S RESPONSIBILITIES (Camera, Pre-processing, & UI)
Viraj is in charge of getting the physical image into the computer, correcting its shape, and displaying the final product.

#### File 1: `src/camera_manager.py`
*   **Class:** `LiveCameraManager`
*   **Goal:** Handle the webcam connection and frame reading.
*   **Methods to write:**
    1.  `__init__(self, camera_index=0)`
        - *Logic:* `self.cap = cv2.VideoCapture(camera_index)`
    2.  `get_frame(self)`
        - *Logic:* `ret, frame = self.cap.read()`
        - *Output:* Returns a NumPy array (the image frame) or `None` if failed.
    3.  `release(self)`
        - *Logic:* `self.cap.release()`

#### File 2: `src/document_detector.py`
*   **Class:** `DocumentDetector`
*   **Goal:** Find the piece of paper in the frame and flatten it (Perspective Warp).
*   **Methods to write:**
    1.  `is_blurry(self, frame, threshold=100)`
        - *Input:* NumPy array (image).
        - *Logic:* `cv2.Laplacian(frame, cv2.CV_64F).var()`.
        - *Output:* Returns `True` if variance < threshold, else `False`.
    2.  `find_document(self, frame)`
        - *Input:* NumPy array (image).
        - *Logic:* Convert to grayscale, Canny edge detection, find contours, find the largest 4-point contour.
        - *Output:* Returns a list of 4 (x, y) coordinates representing the corners of the paper.
    3.  `flatten_document(self, frame, corners)`
        - *Input:* NumPy array (image), list of 4 coordinates.
        - *Logic:* Calculate width/height, use `cv2.getPerspectiveTransform` and `cv2.warpPerspective`.
        - *Output:* Returns a **flattened, perfectly rectangular, top-down NumPy array** of the document. *(This output goes directly to Neel).*

#### File 3: `src/gui.py` & `main.py`
*   **Class:** `LiveDocApp`
*   **Goal:** Build the PyQt6 window to display the camera and the final extracted text.
*   **Integration Logic:** Viraj will import Neel's classes here to complete the pipeline.
    - When the user clicks "Scan", Viraj will pass his flattened image to Neel's `LayoutAnalyzer`.

---

### Part B: NEEL'S RESPONSIBILITIES (AI, Deep Learning, & OCR)
Neel is in charge of taking a perfect top-down image (provided by Viraj), finding the paragraphs/tables, and reading the text.

#### File 1: `src/layout_analyzer.py`
*   **Class:** `LayoutAnalyzer`
*   **Goal:** Use YOLOv11 to find bounding boxes for Text, Titles, and Tables.
*   **Methods to write:**
    1.  `__init__(self, model_path='models/yolo11_doc.pt')`
        - *Logic:* Load the YOLO model: `self.model = YOLO(model_path)`
    2.  `analyze(self, flattened_image)`
        - *Input:* NumPy array (The flattened image passed from Viraj's `flatten_document`).
        - *Logic:* Pass the image to YOLO: `results = self.model(flattened_image)`. Extract the bounding box coordinates and the class names (e.g., "Text", "Table").
        - *Output:* Returns a Python List of dictionaries. Example: 
          `[ {"type": "Title", "box": [x1, y1, x2, y2]}, {"type": "Table", "box": [x1, y1, x2, y2]} ]`

#### File 2: `src/ocr_engine.py`
*   **Class:** `OCREngine`
*   **Goal:** Read the text inside the bounding boxes Neel found in step 1.
*   **Methods to write:**
    1.  `__init__(self)`
        - *Logic:* Initialize PaddleOCR: `self.ocr = PaddleOCR(use_angle_cls=True, lang='en')`
    2.  `extract_from_boxes(self, flattened_image, layout_boxes)`
        - *Input:* The NumPy array image, and the List of dictionaries outputted from `analyze()`.
        - *Logic:* Loop through the `layout_boxes`. For each box, crop the `flattened_image` using numpy slicing: `crop = flattened_image[y1:y2, x1:x2]`. Pass that crop to `self.ocr.ocr(crop)`.
        - *Output:* Returns the fully digitized data! Example:
          `[ {"type": "Title", "text": "Project Report"}, {"type": "Text", "text": "This is the introduction."} ]`

---

## 4. How to Connect The Pipeline (The Grand Finale)

Once Viraj and Neel finish their separate files, they will combine them in `main.py` using this exact pipeline sequence:

```python
# 1. Initialization
camera = LiveCameraManager()
detector = DocumentDetector()
analyzer = LayoutAnalyzer('models/yolo.pt')
ocr = OCREngine()

# 2. Pipeline Execution (Triggered by GUI "Scan" button)
frame = camera.get_frame()

if not detector.is_blurry(frame):
    corners = detector.find_document(frame)
    
    if corners is not None:
        # Viraj's output
        flattened_img = detector.flatten_document(frame, corners)
        
        # Neel's input & processing
        layout_boxes = analyzer.analyze(flattened_img)
        final_text_data = ocr.extract_from_boxes(flattened_img, layout_boxes)
        
        # Viraj displays final_text_data in the GUI!
```

## 5. Git Workflow & Rule of Thumb
1. **Branching:** Viraj writes code in the `Viraj` branch. Neel creates a new branch called `Neel` and writes code there.
2. **Pushing:** Commit changes frequently: `git commit -m "Added DocumentDetector class"`
3. **Merging:** Only merge into `main` when the individual file is tested and works perfectly!
