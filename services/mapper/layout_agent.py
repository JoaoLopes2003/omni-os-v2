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
    is_volatile: bool = Field(
        default=False,
        description="True ONLY if this container holds autonomous live data (video, clock, animation). False for standard UI, including carousels/lists."
    )
    spatial_thought_process: str = Field(
        description="Explicitly state the natural visual boundaries of this container. (e.g., 'Starts below the top nav bar, ends at the left sidebar divider')."
    )
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

    def extract_layout(
        self,
        image_path: str,
        process_name: str,
        is_masked_update: bool = False,
    ) -> tuple[dict, dict]:
        """
        Analyzes a full-screen image and returns macro-containers.
        When `is_masked_update=True`, only maps the unmasked regions of the screenshot.
        """
        with Image.open(image_path) as img:
            real_width, real_height = img.size  # noqa: F841

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
5. VOLATILE CONTAINERS (`is_volatile`):
   Flag containers that are entirely composed of shifting data or autonomous pixels.
   - Example YES (True): Unstructured Data Canvases (code editor text areas, terminal outputs, whiteboard canvases, minimaps views, image/document preview viewers).
   - Example YES (`True`): Autonomous live media (playing video, animated ad banners, live audio waveforms).
   - Example NO (`False`): Static configuration docks, sidebars, or containers holding structural templates (like file lists or media card grids).
6. BOUNDARY ANCHORING:
   Use `spatial_thought_process` to identify the exact visual dividers (lines, background color changes, edges) that frame the container before predicting the 1000-scale coordinates. Containers MUST NOT overlap.

COORDINATE SYSTEM:
Imagine the image is exactly 1000x1000 units.
- [0, 0] is the top-left corner.
- [1000, 1000] is the bottom-right corner.
Output ymin, xmin, ymax, xmax using this 0-1000 scale."""

        if is_masked_update:
            system_instruction += """

7. MASKED INCREMENTAL UPDATE MODE (CRITICAL):
   Parts of this screenshot are covered by dark semi-transparent rectangles labeled `[KNOWN: <container_id>]`.
   - DO NOT map or output any container that is covered by a `[KNOWN: ...]` mask. Those regions are already mapped.
   - ONLY segment the UNMASKED, full-brightness UI region(s) into one or more non-overlapping macro-containers.
   - Use the borders of adjacent `[KNOWN: ...]` masks and natural UI dividers to anchor the exact 0-1000 boundaries of the newly revealed container(s)."""

        prompt = (
            f"Analyze the unmasked regions of this screenshot of '{process_name}'. Output only the newly revealed container(s)."
            if is_masked_update
            else f"Analyze this screenshot of the process '{process_name}'. Output the structured layout."
        )

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

            # Convert 1000-scale to Relative Coordinates (0.0 to 1.0)
            final_containers = []
            for c in raw_data.containers:
                rel_x = round(c.bbox_1000.xmin / 1000.0, 4)
                rel_y = round(c.bbox_1000.ymin / 1000.0, 4)
                rel_w = round((c.bbox_1000.xmax - c.bbox_1000.xmin) / 1000.0, 4)
                rel_h = round((c.bbox_1000.ymax - c.bbox_1000.ymin) / 1000.0, 4)

                final_containers.append({
                    "id": c.id,
                    "description": c.description,
                    "scrollable": c.scrollable,
                    "is_overlay": False,
                    "is_volatile": c.is_volatile,
                    "spatial_thought_process": c.spatial_thought_process,
                    "bbox": {
                        "x": rel_x,
                        "y": rel_y,
                        "w": rel_w,
                        "h": rel_h
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