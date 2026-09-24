import easyocr
import cv2

class OCRProcessor:
    def __init__(self, languages: list[str] = ['en', 'pt'], gpu: bool = True):
        """
        Initialize the EasyOCR reader. 
        Note: If you do not have a dedicated GPU configured with CUDA, set gpu=False.
        """
        print("[OCR] Loading EasyOCR models into memory...")
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def extract_normalized_data(self, image_path: str) -> list[dict]:
        """
        Reads text from an image and normalizes coordinates to a 0-1000 scale.
        
        Returns a list of dictionaries:
        [
            {
                "text": "File",
                "confidence": 0.98,
                "bbox": (xmin, ymin, xmax, ymax)
            }, ...
        ]
        """
        img = cv2.imread(image_path)
        if img is None:
            return []
            
        height, width, _ = img.shape
        results = self.reader.readtext(image_path)
        ocr_data = []
        
        for bbox, text, prob in results:
            # Very low threshold to remove pure static/garbage, but keep UI elements
            if prob < 0.1:
                continue
                
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            
            # Normalize to 0-1000 scale
            xmin = int((min(x_coords) / width) * 1000)
            xmax = int((max(x_coords) / width) * 1000)
            ymin = int((min(y_coords) / height) * 1000)
            ymax = int((max(y_coords) / height) * 1000)
            
            # Format as a list [xmin, ymin, xmax, ymax] for token efficiency
            ocr_data.append({
                "text": text,
                "bbox": [xmin, ymin, xmax, ymax],
                "confidence": round(float(prob), 2)
            })
            
        return ocr_data