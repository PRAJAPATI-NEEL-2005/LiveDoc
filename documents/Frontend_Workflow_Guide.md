# LiveDoc Frontend Workflow Guide (For Neel)

## Welcome to the Frontend!
Viraj has successfully completed the entire AI Backend (Camera, Perspective Warping, YOLO Layout Analysis, and PaddleOCR). Your goal is to build a beautiful, modern Desktop Application using **PyQt6** that ties all of Viraj's backend code together.

## 1. Setup Instructions (Do this first on your Windows Laptop)
Since you are pulling this to your own laptop, you must recreate the exact Python environment Viraj used to prevent AI dependency crashes.

1. Clone the repository and checkout the `Viraj` branch (where the backend code lives).
   ```bash
   git clone https://github.com/PRAJAPATI-NEEL-2005/LiveDoc.git
   cd LiveDoc
   git checkout Viraj
   ```
2. Create a Conda environment strictly locked to **Python 3.10**:
   ```bash
   conda create -n livedoc python=3.10 -y
   conda activate livedoc
   ```
3. Install the heavily-tested, cross-platform dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 2. Your Mission: The GUI (`src/gui.py`)
You need to build a `PyQt6` class (e.g., `LiveDocGUI`) that has the following features:

### A. The Layout
* **Left Panel (Camera Feed):** A large `QLabel` that constantly updates to show the live webcam video.
* **Right Panel (Results):** A `QTextEdit` box that displays the final extracted text.
* **Bottom Panel (Controls):** 
  * A "Start/Stop Camera" button.
  * A "Scan Document" button (Large, prominent).

### B. Integrating Viraj's Backend
You don't need to write any AI code. You just need to import and use Viraj's classes.

**1. Connecting the Camera:**
* Import `LiveCameraManager` from `src.camera_manager`.
* Use a `QTimer` in PyQt6 that ticks every 30 milliseconds.
* Every tick, call `camera.get_frame()`.
* Convert the returned OpenCV frame (BGR) to a PyQt6 `QPixmap` (RGB) and display it on your Left Panel.

**2. Scanning the Document:**
When the user clicks "Scan Document", grab the current frame from the camera and pass it sequentially through the backend:
```python
# 1. Grab the current frame
frame = camera.get_frame()

# 2. Find and flatten the paper
corners = detector.find_document(frame)
if corners is not None:
    flat_paper = detector.flatten_document(frame, corners)
    
    # 3. Find paragraphs and tables (YOLO)
    layout_boxes = layout_analyzer.analyze_document(flat_paper)
    
    # 4. Extract the text
    final_text = ocr_engine.extract_from_boxes(flat_paper, layout_boxes)
    
    # 5. Display the text in your Right Panel!
    text_edit.setPlainText(final_text)
```

---

## 3. The Entry Point (`main.py`)
You need to write the `main.py` file in the root directory. This is the file the user will actually run. It should simply initialize the `QApplication`, create your `LiveDocGUI` window, and start the event loop.

```python
import sys
from PyQt6.QtWidgets import QApplication
from src.gui import LiveDocGUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LiveDocGUI()
    window.show()
    sys.exit(app.exec())
```

## Design Constraints
* **Aesthetics Matter:** Do not make a basic gray Windows 95 box! Use PyQt6 StyleSheets (QSS) to give it a modern dark mode, rounded corners, and nice typography.
* **Threading (Optional but Recommended):** Running PaddleOCR might freeze the GUI for 1-2 seconds. If you want it to feel premium, consider putting the OCR processing inside a `QThread` so the camera feed doesn't freeze while scanning!
