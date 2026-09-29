import os
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Literal
from PIL import Image

# ==========================================
# 1. LLM-Specific Schemas (1000-Scale)
# ==========================================
class BBox1000(BaseModel):
    ymin: int = Field(description="Top edge (0-1000)")
    xmin: int = Field(description="Left edge (0-1000)")
    ymax: int = Field(description="Bottom edge (0-1000)")
    xmax: int = Field(description="Right edge (0-1000)")

class ExtractedContainer(BaseModel):
    id: str = Field(description="Unique snake_case identifier (e.g., 'left_sidebar', 'playback_controls')")
    description: str = Field(description="Semantic description of the container's purpose")
    scrollable: Literal['vertical', 'horizontal', 'both', 'none']
    bbox_1000: BBox1000

class LayoutExtraction(BaseModel):
    view_id: str = Field(description="Unique snake_case name for this entire view (e.g., 'spotify_search_results')")
    description: str = Field(description="What is the user looking at globally?")
    containers: List[ExtractedContainer]

# ==========================================
# 2. The Layout Engine
# ==========================================
class LayoutEngine:
    def __init__(self):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model_name = "gemini-3.1-pro-preview"

    def extract_layout(self, image_path: str, process_name: str) -> tuple[dict, dict]:
        """Analyzes a full-screen image and returns absolute macro-containers."""
        
        # Get actual image dimensions to denormalize the 1000-scale coordinates
        with Image.open(image_path) as img:
            real_width, real_height = img.size

        system_instruction = """You are the Omni-OS Visual Layout Engine. 
Your task is to analyze an application GUI and segment it into non-overlapping macro-containers.

RULES FOR SEGMENTATION:
1. Divide the UI into major structural blocks: sidebars, top navigation bars, and playback controls. 
   - CRITICAL: Do not wrap the entire "Main Content Area" into a single massive container. Slice the main content into logical horizontal or vertical sections (e.g., separate a 'Hero Banner' section from a 'Grid Feed' section below it).
2. DO NOT map individual tiny buttons. Map the CONTAINERS that hold them.
3. Completely cover the application UI, but containers MUST NOT overlap unless one is a floating overlay/modal.
4. SCROLLABILITY & INHERITANCE (CRITICAL): Mark scrollable as 'vertical', 'horizontal', 'both', or 'none'.
    - Base Rule: Flag it if you see a scrollbar, cut-off content, or if UI conventions strongly imply it (e.g., a main application feed, a sidebar library, or a carousel). Trust your knowledge of standard app layouts.
    - INHERITANCE: Because you are slicing large scrollable pages into smaller chunks, these chunks MUST inherit the page's scrollability. If the overall main content area scrolls vertically, EVERY container within it (e.g., the Hero Banner, Quick Links) must be marked at least 'vertical', because scrolling the mouse over them moves the page.
    - MULTI-AXIS: If a chunk internally scrolls horizontally (like a media carousel) AND sits inside a vertically scrollable page, mark it 'both'.

COORDINATE SYSTEM:
Imagine the image is exactly 1000x1000 units.
- [0, 0] is the top-left corner.
- [1000, 1000] is the bottom-right corner.
Output ymin, xmin, ymax, xmax using this 0-1000 scale."""

        prompt = f"Analyze this screenshot of the process '{process_name}'. Output the structured layout."
        
        uploaded_file = self.client.files.upload(file=image_path)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[uploaded_file, prompt],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=LayoutExtraction
                )
            )
            
            raw_data = response.parsed
            
            # Convert 1000-scale to Absolute OS Coordinates (x, y, w, h)
            final_containers = []
            for c in raw_data.containers:
                # Math: (normalized_val / 1000) * real_dimension
                x = int((c.bbox_1000.xmin / 1000.0) * real_width)
                y = int((c.bbox_1000.ymin / 1000.0) * real_height)
                x2 = int((c.bbox_1000.xmax / 1000.0) * real_width)
                y2 = int((c.bbox_1000.ymax / 1000.0) * real_height)
                
                final_containers.append({
                    "id": c.id,
                    "description": c.description,
                    "scrollable": c.scrollable,
                    "is_overlay": False, # Layout engine maps base UI; overlays are temporal
                    "bbox": {
                        "x": x,
                        "y": y,
                        "w": x2 - x,
                        "h": y2 - y
                    },
                    "elements": [],
                    "items": []
                })

            final_view = {
                "view_id": raw_data.view_id,
                "process_name": process_name,
                "description": raw_data.description,
                "containers": final_containers
            }
            
            token_usage = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count
            }

            return final_view, token_usage

        finally:
            self.client.files.delete(name=uploaded_file.name)