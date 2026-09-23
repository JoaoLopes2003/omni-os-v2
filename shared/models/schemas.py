from pydantic import BaseModel

class BoundingBox(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def to_absolute(self, screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
        return (
            int(self.xmin * screen_width),
            int(self.ymin * screen_height),
            int(self.xmax * screen_width),
            int(self.ymax * screen_height)
        )

    def get_absolute_center(self, screen_width: int, screen_height: int) -> tuple[int, int]:
        abs_xmin, abs_ymin, abs_xmax, abs_ymax = self.to_absolute(screen_width, screen_height)
        center_x = (abs_xmin + abs_xmax) // 2
        center_y = (abs_ymin + abs_ymax) // 2
        return center_x, center_y

class UIElement(BaseModel):
    element_id: int
    bbox: BoundingBox
    text: str
    confidence: float

if __name__ == "__main__":
    # Create a dummy bounding box (representing the middle 20% of the screen)
    dummy_box = BoundingBox(xmin=0.4, ymin=0.4, xmax=0.6, ymax=0.6)
    print(f"Normalized schema:\n{dummy_box}\n")
    
    # Convert to absolute pixels for a standard 1080p display
    screen_w, screen_h = 1920, 1080
    absolute_coords = dummy_box.to_absolute(screen_w, screen_h)
    
    print(f"Absolute pixels on a {screen_w}x{screen_h} screen:")
    print(f"(xmin, ymin, xmax, ymax) -> {absolute_coords}")