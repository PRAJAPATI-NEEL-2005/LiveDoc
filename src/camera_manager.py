import sys
import cv2


def _get_capture_backend():
    """Use DirectShow for fast opening on Windows; default on macOS/Linux."""
    return cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY


class LiveCameraManager:
    def __init__(self, camera_index=None):
        """Initialize the webcam connection. If camera_index is None, camera is not opened yet."""
        self.camera_index = camera_index
        self.cap = None
        if camera_index is not None:
            self.open(camera_index)
        else:
            self.cap = cv2.VideoCapture()

    def is_opened(self):
        """Check if the camera is currently connected and open."""
        return self.cap is not None and self.cap.isOpened()

    def open(self, camera_index=None):
        """Open a video capture device."""
        if camera_index is not None:
            self.camera_index = camera_index
        if self.camera_index is None:
            self.camera_index = 0

        self.release()
        backend = _get_capture_backend() if isinstance(self.camera_index, int) else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.camera_index, backend)
        if not self.cap.isOpened():
            # Fallback to CAP_ANY if platform-specific backend fails
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_ANY)

        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}.")
            return False
        return True

    @staticmethod
    def detect_cameras(max_index=6):
        """
        Scan camera indices 0 through max_index-1 and return a list of
        available cameras as (index, name) tuples.
        
        On Windows, cv2.VideoCapture only gives us an index, not a friendly
        name, so we label them 'Camera 0', 'Camera 1', etc. The user can
        identify their phone / external camera by trying each one.
        """
        backend = _get_capture_backend()
        available = []
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                # Read one frame to confirm it's a real camera, not a stale handle
                ret, _ = cap.read()
                if ret:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    label = f"Camera {idx}  ({w}x{h})"
                    available.append((idx, label))
                cap.release()
            else:
                # On macOS, camera indices are contiguous; break early if not found
                if sys.platform != "win32" and idx > 0:
                    break
        return available

    def switch_camera(self, camera_index):
        """Release the current camera and open a different one."""
        return self.open(camera_index)

    def get_frame(self):
        """Capture and return a single frame from the webcam."""
        if self.cap is None or not self.cap.isOpened():
            return None
            
        ret, frame = self.cap.read()
        if ret and frame is not None:
            return frame
        else:
            return None
            
    def release(self):
        """Release the webcam resource."""
        if self.cap is not None:
            try:
                if self.cap.isOpened():
                    self.cap.release()
            except Exception:
                pass
            self.cap = cv2.VideoCapture()


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
