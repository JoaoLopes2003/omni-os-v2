import easyocr
import numpy as np
import time
from src.core.capture import ScreenCaptureEngine

class TextPerceptionEngine:
    def __init__(self, languages: list[str] = ['en', 'pt'], gpu: bool = True):
        """
        Initialize the EasyOCR reader. 
        Note: If you do not have a dedicated GPU configured with CUDA, set gpu=False.
        """
        print("Loading OCR weights...")
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def extract_text(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> list[dict]:
        """
        Extracts text from an RGB numpy array.
        
        Returns a list of dictionaries:
        [
            {
                "text": "File",
                "confidence": 0.98,
                "bbox": (xmin, ymin, xmax, ymax)
            }, ...
        ]
        """
        # Pass the frame to the reader
        results = self.reader.readtext(frame)

        # Loop through the results and filter those below the confidence threshold
        results = [r for r in results if r[2] > confidence_threshold]

        # Convert the 4 polygon points into xmin, ymin, xmax, ymax
        bboxs = []
        for coords, text, conf in results:
            xs = [point[0] for point in coords]
            ys = [point[1] for point in coords]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            bboxs.append({
                'text': text,
                'confidence': conf,
                'bbox': (xmin, ymin, xmax, ymax)
            })

        return bboxs

if __name__ == "__main__":
    # Test block
    
    # Instantiate the ScreenCaptureEngine
    engine = ScreenCaptureEngine()
    engine.grab_frame()

    # Instantiate TextPerceptionEngine
    reader = TextPerceptionEngine()

    # Start time.perf_counter()
    start_time = time.perf_counter()

    # Pass the frame to extract_text()
    frame = engine.grab_frame()
    bboxs = reader.extract_text(frame)

    # End time.perf_counter()
    end_time = time.perf_counter()
    
    # Print the first 5 results to verify your coordinate math
    print(bboxs[:5])

    # Print the elapsed time in milliseconds
    latency_ms = (end_time - start_time) * 1000
    print(f"It took {latency_ms} ms to process a frame.")

    pass