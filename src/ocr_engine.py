from paddleocr import PaddleOCR
import numpy as np
import cv2

class OCREngine:
    def __init__(self, lang='en'):
        """
        Initialize the PaddleOCR engine.
        use_angle_cls=True automatically rotates text if it's sideways or upside down.
        show_log=False stops Paddle from printing massive logs to the terminal.
        """
        print("Initializing PaddleOCR (This might take a moment to download models the first time)...")
        # We use use_angle_cls=True to automatically fix sideways/upside-down text
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        print("PaddleOCR Initialized successfully!")

    def extract_full_text(self, image):
        """
        Extracts all text from the given image array.
        Returns a single combined string.
        """
        # Run OCR on the image
        results = self.ocr.ocr(image, cls=True)
        extracted_lines = []
        
        # Check if text was actually found
        if not results or not results[0]:
            return ""
            
        # Extract just the text strings from PaddleOCR's complex nested output
        for line in results[0]:
            text = line[1][0]  # line[1][0] contains the actual string
            extracted_lines.append(text)
            
        return "\n".join(extracted_lines)

    def extract_from_boxes(self, image, layout_boxes):
        """
        Extracts text specifically from bounding boxes provided by the LayoutAnalyzer.
        """
        final_data = []
        
        for box in layout_boxes:
            x1, y1, x2, y2 = box["box"]
            box_type = box["type"]
            
            # Crop the image array using numpy slicing
            crop = image[y1:y2, x1:x2]
            
            # Skip if the crop is invalid (e.g., width or height is 0)
            if crop.size == 0:
                continue
                
            # Run OCR on just this specific cropped block
            text = self.extract_full_text(crop)
            
            final_data.append({
                "type": box_type,
                "text": text
            })
            
        return final_data

# --- Quick Test ---
if __name__ == "__main__":
    # Create a quick dummy image to test OCR locally
    test_img = np.zeros((200, 500, 3), dtype=np.uint8)
    test_img.fill(255) # White background
    cv2.putText(test_img, "LiveDoc Backend Test!", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    print("Testing OCREngine...")
    engine = OCREngine()
    
    print("Extracting text from dummy image...")
    text = engine.extract_full_text(test_img)
    print("---------------------------------")
    print(f"Extracted Text:\n{text}")
    print("---------------------------------")
