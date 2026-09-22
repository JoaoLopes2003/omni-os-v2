import time
import mss
import numpy as np
import cv2

class ScreenCaptureEngine:
    def __init__(self, monitor_index: int = 1):
        """
        Initialize the MSS capture instance and define the target monitor.
        """
        self.sct = mss.MSS()
        self.monitor = self.sct.monitors[monitor_index]

    def grab_frame(self) -> np.ndarray:
        """
        Captures the screen and returns an RGB numpy array (Height, Width, 3).
        """
        raw_data = self.sct.grab(self.monitor)
        frame = np.array(raw_data, dtype=np.uint8)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        return rgb
    
    def close(self):
        """
        Clean up the mss instance.
        """
        self.sct.close()

if __name__ == "__main__":
    # Instantiate the ScreenCaptureEngine
    engine = ScreenCaptureEngine()
    
    # Warm-up (run grab_frame once without timing it, as the first call usually has overhead)
    engine.grab_frame()
    
    # Start time.perf_counter()
    start_time = time.perf_counter()

    # Grab a frame
    frame = engine.grab_frame()

    # End time.perf_counter()
    end_time = time.perf_counter()
    
    # Print the shape of the array (Should be H, W, 3)
    print(f"Frame shape: {frame.shape}")

    # Print the elapsed time in milliseconds
    latency_ms = (end_time - start_time) * 1000
    print(f"It took {latency_ms} ms to grab a frame.")