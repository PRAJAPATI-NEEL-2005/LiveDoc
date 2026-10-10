from ultralytics import YOLO
import cv2
import os

class LayoutAnalyzer:
    def __init__(self, model_path='models/yolo11_doc.pt'):
        """
        Load the trained YOLO model for document layout analysis.
        If the model isn't downloaded yet, it defaults to a fallback mode.
        """
        self.use_yolo = False
        
        # Check if we have actually trained and downloaded the YOLO weights yet
        if os.path.exists(model_path):
            print(f"Loading YOLO model from {model_path}...")
            self.model = YOLO(model_path)
            self.use_yolo = True
            print("YOLO model loaded successfully!")
        else:
            print(f"WARNING: YOLO model not found at {model_path}.")
            print("Falling back to standard full-page analysis for now.")
            
    def analyze(self, image):
        """
        Analyzes the flattened document to find Titles, Text Blocks, and Tables.
        Returns a list of dictionaries. 
        Example: [{"type": "Text", "box": [x1, y1, x2, y2]}]
        """
        if self.use_yolo:
            return self._yolo_analyze(image)
        else:
            return self._fallback_analyze(image)
            
    def _yolo_analyze(self, image):
        """
        Real AI mode using YOLO bounding boxes.
        """
        # Run YOLO inference
        results = self.model(image)[0]
        layout_boxes = []
        
        for box in results.boxes:
            # Extract box coordinates (Top-Left X, Y, Bottom-Right X, Y)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Extract class name (e.g. "Title", "Text", "Table")
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            
            layout_boxes.append({
                "type": class_name,
                "box": [x1, y1, x2, y2]
            })
            
        return layout_boxes
        
    def _fallback_analyze(self, image):
        """
        Fallback mode if the user hasn't trained the YOLO model yet.
        This just returns the entire flattened page as a single "Text" block
        so the OCR engine can still run and read the whole page.
        """
        h, w = image.shape[:2]
        # Treat the entire page as one big text block
        return [{"type": "Text", "box": [0, 0, w, h]}]

# --- Quick Test ---
if __name__ == "__main__":
    import numpy as np
    print("Testing LayoutAnalyzer...")
    
    analyzer = LayoutAnalyzer()
    
    # Create a dummy image
    test_img = np.zeros((800, 600, 3), dtype=np.uint8)
    boxes = analyzer.analyze(test_img)
    
    print("---------------------------------")
    print(f"Layout Boxes Found: {boxes}")
    print("---------------------------------")
