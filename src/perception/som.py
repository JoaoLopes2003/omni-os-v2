import cv2
import numpy as np
from src.core.schemas import UIElement, BoundingBox

class SomRenderer:
    def __init__(self):
        """
        Initializes the Set-of-Marks renderer with standard visual configurations.
        Note: OpenCV uses BGR color order by default for drawing, but since 
        our frame is already RGB, we define our colors in RGB.
        """
        self.box_color = (0, 255, 0)   # Green boxes
        self.text_color = (255, 0, 0)  # Red text
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2

    def draw_marks(self, frame: np.ndarray, elements: list[UIElement]) -> np.ndarray:
        """
        Takes an RGB frame and a list of UIElements, drawing bounding boxes 
        and numeric IDs on a copy of the frame.
        """
        # Always work on a copy so we don't mutate the original frame in memory
        annotated_frame = frame.copy()
        
        # Extract screen_height and screen_width from annotated_frame.shape
        # Remember: shape returns (height, width, channels)
        screen_height, screen_width, _ = annotated_frame.shape
        
        # Loop through the 'elements' list
        for element in elements:
            # Call element.bbox.to_absolute() with your width/height to get integer pixels
            xmin, ymin, xmax, ymax = element.bbox.to_absolute(screen_width, screen_height)
            
            # Draw the bounding box using cv2.rectangle()
            start_point = (xmin, ymin)
            end_point = (xmax, ymax)
            annotated_frame = cv2.rectangle(
                annotated_frame, start_point, end_point, self.box_color, self.thickness
            )
            
            # Draw the element_id using cv2.putText()
            annotated_frame = cv2.putText(
                annotated_frame,
                f"[ {element.element_id} ]",
                (start_point[0], start_point[1] - 10),
                self.font,
                self.font_scale,
                self.text_color,
                self.thickness
            )
            
        return annotated_frame

if __name__ == "__main__":
    # --- Dummy Test Block ---
    
    # Create a blank dark grey "screen" (1080p)
    dummy_frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 50 
    
    # Create a dummy UI Element using our normalized schema
    dummy_element = UIElement(
        element_id=1,
        bbox=BoundingBox(xmin=0.4, ymin=0.4, xmax=0.6, ymax=0.6),
        text="Dummy Button",
        confidence=0.99
    )
    
    # Instantiate renderer and draw
    renderer = SomRenderer()
    result_frame = renderer.draw_marks(dummy_frame, [dummy_element])
    
    # Save to disk to verify your math and drawing logic
    # Note: cv2.imwrite expects BGR, so we temporarily convert our RGB frame back to BGR just for saving
    cv2.imwrite("som_debug.png", cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))
    print("Saved som_debug.png to your project directory. Open it to check your work.")