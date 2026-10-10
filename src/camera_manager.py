import cv2

class LiveCameraManager:
    def __init__(self, camera_index=0):
        """Initialize the webcam connection."""
        self.cap = cv2.VideoCapture(camera_index)
        
        # Check if the camera opened successfully
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_index}.")
            
    def get_frame(self):
        """Capture and return a single frame from the webcam."""
        if not self.cap.isOpened():
            return None
            
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            return None
            
    def release(self):
        """Release the webcam resource."""
        if self.cap.isOpened():
            self.cap.release()

# --- Quick Test ---
# This block only runs if you execute this specific file directly
if __name__ == "__main__":
    print("Testing LiveCameraManager... Press 'q' to quit.")
    camera = LiveCameraManager(0)
    
    while True:
        frame = camera.get_frame()
        if frame is None:
            print("Failed to grab frame.")
            break
            
        cv2.imshow("LiveDoc - Camera Test", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    camera.release()
    cv2.destroyAllWindows()
