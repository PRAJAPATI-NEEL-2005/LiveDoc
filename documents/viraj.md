# Viraj's Work Log: LiveDoc Project

This document tracks all the work completed by Viraj on the **LiveDoc: Real-Time Intelligent Document Digitizer** project. It serves as a record of methods used, problems solved, and files created so that Neel and other team members can easily understand the progress.

---

## [2026-10-10] - Initial Setup & Camera Integration

### 1. Project Initialization & Folder Structure
- **What was done:** Set up the foundational directory structure for the entire project.
- **Method:** Created `src/` for OOP classes, `models/` for YOLO weights, and `documents/` for documentation. Created empty template files for both Viraj and Neel (`camera_manager.py`, `document_detector.py`, `layout_analyzer.py`, `ocr_engine.py`, `gui.py`).
- **Why:** To ensure a clean separation of concerns and maintain a proper OOP architecture from day one.

### 2. Dependency Management & Environment Setup
- **What was done:** Created `requirements.txt` and successfully configured the Python environment after surviving extreme dependency hell.
- **Method:** 
  - Defined dependencies: `opencv-python<5`, `ultralytics`, `paddlepaddle==2.6.2`, `paddleocr<=2.8.1`, `PyQt6`, `numpy<2`.
  - **Critical Fix 1 (NumPy):** PyTorch and PaddleOCR crashed on `NumPy 2.x`. We locked `numpy<2`.
  - **Critical Fix 2 (OpenCV):** `opencv-python` version 5 requires `NumPy 2.x`, creating a fatal conflict. We downgraded to `opencv-python<5`.
  - **Critical Fix 3 (PaddleOCR Mac Crash):** The newest `PaddlePaddle 3.0-beta` has a deep C++ bug on Mac ARM processors when loading the `textline_orientation` model (`ValueError: strides is not right`). We downgraded to the incredibly stable `paddlepaddle==2.6.2` and `paddleocr<=2.8.1`.
- **Why:** By locking exactly these stable versions in `requirements.txt`, we ensure 100% cross-platform compatibility. Neel can pull this code on his Windows laptop and skip all these errors entirely!

### 3. Camera Integration (`src/camera_manager.py`)
- **What was done:** Implemented the `LiveCameraManager` class to handle real-time webcam feeds.
- **Method:** 
  - Used OpenCV (`cv2.VideoCapture(0)`) to access the system's primary webcam.
  - Implemented `get_frame()` to continuously read frames as NumPy arrays.
  - Added a quick-test block at the bottom of the script using `cv2.imshow` so the camera feed can be tested instantly by running `python src/camera_manager.py`.
- **Why:** This is the first critical step of the pipeline. Neel's AI models will rely on these NumPy array frames to perform layout analysis and OCR.

### 4. Advanced Document Detection & Perspective Warping (`src/document_detector.py`)

- **What was done:** Designed and implemented the `DocumentDetector` class to automatically locate a physical piece of paper in a live, noisy camera feed, crop it, and geometrically flatten it into a perfect top-down view.
- **Why we need this (The Problem Statement):** A raw camera feed contains immense background noise (desks, hands, lighting shadows, walls). If we feed this directly to Neel's AI models (YOLO / OCR), they will fail because the text is often small, skewed by the camera angle, and surrounded by irrelevant objects. This class acts as a crucial pre-processing pipeline that completely isolates the region of interest (the paper) and mathematically straightens the text. This drastically increases the accuracy of our OCR engine.

**Deep Dive: How the Computer Vision Pipeline Works (Function by Function)**

#### 1. `is_blurry(self, frame, threshold=50)` - Blur Detection
- **Objective:** Prevent the pipeline from processing frames where the user is moving the camera or paper too fast, which would result in failed OCR.
- **The Math & Logic:** 
  - First, we convert the image to Grayscale (`cv2.cvtColor`). We don't need color to find edges, and dropping color channels speeds up processing.
  - We use the **Laplacian operator** (`cv2.Laplacian(gray, cv2.CV_64F)`). In image processing, the Laplacian measures the 2nd derivative of the image. It highlights regions of rapid intensity change (i.e., sharp edges).
  - We calculate the **variance** (`.var()`) of the Laplacian. An image with high variance has a wide spread of sharp edges (in focus). An image with low variance has very few sharp edges (out of focus or completely blank).
  - We compare this variance against a `threshold=50`. If it's lower, we classify the frame as "BLURRY". *(Note: Pointing the camera at a blank wall will return blurry because there are no edges!)*

#### 2. `find_document(self, frame)` - Edge & Contour Extraction
- **Objective:** Find the 4 mathematical coordinate points `(x,y)` of the paper in the real world.
- **The Logic (The OpenCV Pipeline):**
  - **Bilateral Filtering (`cv2.bilateralFilter`):** We apply a filter with parameters `(9, 75, 75)`. Unlike a normal Gaussian blur which blurs everything, a bilateral filter is edge-preserving. It smooths out the textures of the desk/background but keeps the hard edges of the white paper perfectly sharp.
  - **Canny Edge Detection (`cv2.Canny(blurred, 50, 150)`):** This algorithm looks for strong gradients (sudden color changes, like white paper against a dark desk) and draws them as thin white outlines on a black canvas.
  - **Morphological Closing (`cv2.morphologyEx`):** Because of lighting glares, the outline of the paper might have tiny "gaps" or "breaks". We create a `5x5` rectangular kernel and apply a morphological "close" (a dilation followed by an erosion). This essentially connects broken dots along the paper's border to form a solid, unbroken outline.
  - **Contour Finding (`cv2.findContours`):** We tell OpenCV to find all enclosed shapes in our edge map. We sort them by area (largest to smallest) and only keep the top 5 largest shapes.
  - **Area Filtering:** We explicitly ignore any shape smaller than `10,000` pixels. This ensures we don't accidentally detect a phone, a hand, or a small shadow.
  - **Polygonal Approximation (`cv2.approxPolyDP`):** For each large shape, we calculate its perimeter (`cv2.arcLength`). We then ask OpenCV to approximate the shape using fewer points (allowing a 2% margin of error `0.02 * peri`).
  - **The Final Check:** If the approximated shape has exactly `4` points (a quadrilateral) and is convex (bulges outward, no indented corners), we confidently declare it as our document!

#### 3. `_order_points(self, pts)` - Coordinate Standardization
- **Objective:** The 4 points we found in the previous step could be in any random order. To flatten the image properly, the math requires them to be strictly ordered: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
- **The Math:**
  - **Top-Left / Bottom-Right:** We add the `x + y` coordinates. The smallest sum is the Top-Left corner (closest to `0,0`). The largest sum is the Bottom-Right.
  - **Top-Right / Bottom-Left:** We subtract the coordinates `x - y` (or `y - x`). The minimum difference gives us the Top-Right, and the maximum difference gives us the Bottom-Left.

#### 4. `flatten_document(self, frame, corners)` - Perspective Warping
- **Objective:** Take a skewed, angled photo of a document and geometrically warp it so it looks like it was scanned completely flat on a glass scanner.
- **The Math:**
  - We use the Euclidean distance formula `sqrt((x2-x1)^2 + (y2-y1)^2)` to calculate the exact width and height of the angled paper in pixels.
  - We create a hypothetical "destination" matrix (`dst`) representing a perfectly flat rectangle with those exact widths and heights starting at `[0,0]`.
  - **`cv2.getPerspectiveTransform(rect, dst)`**: This computes a 3x3 transformation matrix. It mathematically calculates exactly how to stretch, pull, and skew the pixels from the angled camera view to match our flat destination view.
  - **`cv2.warpPerspective`**: This function applies the 3x3 matrix to the actual image pixels, effectively cropping the background away and "flattening" the paper into our final, perfect output.

---

### 5. Layout Analysis (`src/layout_analyzer.py`)
- **What was done:** Wrote the `LayoutAnalyzer` class using YOLOv11 to find exactly *where* text and tables are located inside the flattened document.
- **Why we need this:** A raw OCR engine just reads text left-to-right, top-to-bottom. If a document has two columns, the OCR will read straight across both columns, creating complete gibberish. YOLO solves this by drawing a bounding box around Column 1 and Column 2 separately.
- **How it works:**
  - `__init__`: Tries to load the trained PyTorch weights (`yolo11_doc.pt`). If the weights haven't been trained yet, it catches the error and safely falls back to treating the entire page as one big text block.
  - `_yolo_analyze`: Feeds the NumPy image to the neural network. The network outputs a list of `boxes`. It extracts `[x1, y1, x2, y2]` (the coordinates of the box) and the `class_name` (e.g., "Title", "Text", "Table"). It returns this structured data as a list of Python dictionaries.

### 6. Intelligent Text Extraction (`src/ocr_engine.py`)
- **What was done:** Wrote the `OCREngine` class using Baidu's PaddleOCR to actually read the words on the page.
- **Why we need this:** We need the industry's best open-source OCR engine. PaddleOCR was chosen because it supports `use_angle_cls=True`, which means if a user holds a document sideways or upside down, the AI automatically detects the text orientation and flips it before reading!
- **How it works:**
  - `extract_full_text(image)`: Passes a NumPy array to the PaddleOCR neural network. Paddle returns a massively nested list containing boxes, coordinates, strings, and confidence scores. This function loops through that nightmare of a list, extracts *only* the string text (`line[1][0]`), and joins them all with a newline `\n`.
  - `extract_from_boxes(image, layout_boxes)`: This is the genius function. It takes the output from YOLO (a list of box coordinates) and uses NumPy array slicing (`image[y1:y2, x1:x2]`) to physically crop the image down to *just* that specific paragraph or table. It then passes that tiny cropped image to `extract_full_text`. This guarantees column text and tables are read perfectly.

---

*Note for Neel: Viraj has completed the AI Backend! You are now fully cleared to build `gui.py` and `main.py` to tie it all together.*
