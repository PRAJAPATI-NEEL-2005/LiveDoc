import cv2
import numpy as np

class DocumentDetector:
    def __init__(self):
        """Initialize the document detector."""
        pass

    def is_blurry(self, frame, threshold=50):
        """
        Check if the image is too blurry to read.
        Returns True if blurry, False if sharp.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold

    def find_document(self, frame):
        """
        Find the 4 corners of the document in the frame.
        Returns a numpy array of the 4 coordinates, or None if no document is found.
        """
        # Store frame area to ensure the document takes up a decent chunk of the screen
        frame_area = frame.shape[0] * frame.shape[1]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Heavy blur to remove text inside the paper, leaving only the outer paper edge
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        
        # Adaptive thresholding often works better than Canny in weird lighting
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operation to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        edged = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Save edged for debugging in the test block
        self.debug_edges = edged.copy()
        
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        for c in contours:
            area = cv2.contourArea(c)
            # The document must take up at least 5% of the frame and not the entire frame
            if area < (frame_area * 0.05) or area > (frame_area * 0.95):
                continue
                
            # Looser approximation (0.05 instead of 0.02) to handle slightly bent paper
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.05 * peri, True)
            
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx.reshape(4, 2)
                
        return None

    def _order_points(self, pts):
        """
        Order points consistently: [top-left, top-right, bottom-right, bottom-left].
        """
        rect = np.zeros((4, 2), dtype="float32")
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect

    def flatten_document(self, frame, corners):
        """
        Warp the perspective to flatten the document into a top-down view.
        Returns the cropped and flattened document image as a numpy array.
        """
        rect = self._order_points(corners)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
        return warped

# --- Quick Test ---
if __name__ == "__main__":
    from camera_manager import LiveCameraManager
    
    print("Testing DocumentDetector... Show a piece of paper to the camera!")
    print("Press 'q' to quit.")
    
    camera = LiveCameraManager(0)
    detector = DocumentDetector()
    
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
            
        display_frame = frame.copy()
        
        if detector.is_blurry(frame):
            cv2.putText(display_frame, "BLURRY!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        corners = detector.find_document(frame)
        
        # Show what the computer is actually "seeing" to help debug!
        if hasattr(detector, 'debug_edges'):
            cv2.imshow("Debug: What the computer sees (Edges)", detector.debug_edges)
        
        if corners is not None:
            cv2.drawContours(display_frame, [corners], -1, (0, 255, 0), 3)
            cv2.putText(display_frame, "Document Found", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            flattened = detector.flatten_document(frame, corners)
            cv2.imshow("Flattened Document", flattened)
            
        cv2.imshow("Live Scanner", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    camera.release()
    cv2.destroyAllWindows()
