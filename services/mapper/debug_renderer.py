import cv2
import os

class DebugRenderer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.thickness = 1

    def draw_ocr_bboxes(self, image_path: str, ocr_data: list[dict], output_path: str):
        """Draws bounding boxes around EasyOCR detections."""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[Renderer] Could not load image: {image_path}")
            return
            
        h, w, _ = frame.shape
        
        for item in ocr_data:
            bbox = item.get("bbox")
            text = item.get("text", "")
            if bbox and len(bbox) == 4:
                # Denormalize (0-1000) -> absolute pixels
                xmin = int((bbox[0] / 1000) * w)
                ymin = int((bbox[1] / 1000) * h)
                xmax = int((bbox[2] / 1000) * w)
                ymax = int((bbox[3] / 1000) * h)
                
                # Draw green box and red text
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, text, (xmin, max(0, ymin - 5)), self.font, self.font_scale, (0, 0, 255), self.thickness)
                
        cv2.imwrite(output_path, frame)

    def draw_gemini_elements(self, image_path: str, elements: list, output_path: str):
        """Draws red dots at the center coordinates determined by Gemini."""
        frame = cv2.imread(image_path)
        if frame is None:
            return
            
        h, w, _ = frame.shape
        
        for element in elements:
            cx_norm = element.get("center_x", -1)
            cy_norm = element.get("center_y", -1)
            el_type = element.get("element_type", "unknown")
            
            # Use the ID or text for the label
            label = element.get("id", element.get("text", "unnamed"))
            
            if cx_norm != -1 and cy_norm != -1:
                cx = int((cx_norm / 1000) * w)
                cy = int((cy_norm / 1000) * h)
                
                # Draw red dot
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                
                # Draw blue label next to the dot
                display_text = f"{label} ({el_type})"
                cv2.putText(frame, display_text, (cx + 8, cy + 4), self.font, self.font_scale, (255, 0, 0), self.thickness)
                
        cv2.imwrite(output_path, frame)