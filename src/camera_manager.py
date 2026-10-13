import cv2

class LiveCameraManager:
    def __init__(self, camera_index=0):
        """Initialize the webcam connection."""
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index)
        
        # Check if the camera opened successfully
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_index}.")
            
    @staticmethod
    def detect_cameras(max_index=10):
        """
        Scan camera indices 0 through max_index-1 and return a list of
        available cameras as (index, name) tuples.
        
        On Windows, cv2.VideoCapture only gives us an index, not a friendly
        name, so we label them 'Camera 0', 'Camera 1', etc.  The user can
        identify their Smart Connect / phone camera by trying each one.
        """
        available = []
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)   # DirectShow is faster on Windows
            if cap.isOpened():
                # Read one frame to confirm it's a real camera, not a stale handle
                ret, _ = cap.read()
                if ret:
                    # Try to grab a human-readable backend name
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    label = f"Camera {idx}  ({w}x{h})"
                    available.append((idx, label))
                cap.release()
        return available

    def switch_camera(self, camera_index):
        """Release the current camera and open a different one."""
        self.release()
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_index}.")
            return False
        return True

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
