import os
import tempfile
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from PIL import Image

# ==========================================
# 1. LLM-Specific Schemas (1000-Scale)
# ==========================================
class LLMElement(BaseModel):
    id: str = Field(description="Unique snake_case identifier")
    text: Optional[str] = Field(default=None, description="Visible text on the element, if any")
    element_type: Literal['button', 'icon', 'input', 'text', 'image', 'dropdown']
    description: str = Field(description="Semantic description of the element")

    # Forces the model to determine alignment
    text_align: Optional[Literal['left', 'center', 'right']] = Field(
        default=None, 
        description="For text/input elements, specify the horizontal text alignment."
    )
    
    center_x_1000: int = Field(description="Center X coordinate (0-1000) relative to the provided image bounds")
    center_y_1000: int = Field(description="Center Y coordinate (0-1000) relative to the provided image bounds")
    
    is_dynamic: bool = Field(default=False, description="True if content changes often (e.g., currently playing song)")
    opens_container: Optional[str] = Field(default=None, description="ID of popup/menu it opens")
    navigates_to_view: Optional[str] = Field(default=None, description="ID of new screen it navigates to")

class LLMTemplate(BaseModel):
    template_id: str
    description: str
    elements: List[LLMElement]

class LLMCollectionItem(BaseModel):
    index: int
    ymin: int = Field(description="Top edge (0-1000)")
    xmin: int = Field(description="Left edge (0-1000)")
    ymax: int = Field(description="Bottom edge (0-1000)")
    xmax: int = Field(description="Right edge (0-1000)")

class ContainerExtraction(BaseModel):
    thought_process: str = Field(
        description="Briefly describe the layout. Are these vertical list rows or grid cards? Explicitly state that bounding boxes must span the full width of the text/content before outputting coordinates."
    )
    elements: List[LLMElement] = Field(default_factory=list)
    item_template: Optional[LLMTemplate] = Field(default=None)
    items: List[LLMCollectionItem] = Field(default_factory=list)

# ==========================================
# 2. The Element Engine
# ==========================================
class ElementEngine:
    def __init__(self):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model_name = "gemini-3.1-pro-preview"
        self.max_aspect_ratio = 3.0  # Safe boundary for Vision Transformers

    def extract_elements(
        self, 
        cropped_image_path: str, 
        container_id: str, 
        container_description: str
    ) -> tuple[str, dict, dict]:
        """Analyzes a cropped container using dynamic padding for extreme aspect ratios."""
        
        with Image.open(cropped_image_path) as img:
            crop_width, crop_height = img.size
            
            target_width = crop_width
            target_height = crop_height
            
            # Check aspect ratio and calculate dynamic padding
            if crop_width / crop_height > self.max_aspect_ratio:
                # Too wide (e.g., taskbar). Pad the height.
                target_height = int(crop_width / self.max_aspect_ratio)
            elif crop_height / crop_width > self.max_aspect_ratio:
                # Too tall (e.g., vertical dock). Pad the width.
                target_width = int(crop_height / self.max_aspect_ratio)
                
            needs_padding = (target_width != crop_width) or (target_height != crop_height)
            
            if needs_padding:
                padded_img = Image.new("RGB", (target_width, target_height), (0, 0, 0))
                padded_img.paste(img, (0, 0))
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    padded_img.save(temp_file, format="PNG")
                    process_image_path = temp_file.name
            else:
                process_image_path = cropped_image_path

        padding_notice = ""
        if needs_padding:
            padding_notice = """
IMAGE PADDING NOTICE:
This UI element was padded with a solid black background to preserve resolution. 
Only map the UI elements inside the actual application crop located in the top-left corner. Ignore the black void completely.
"""

        system_instruction = f"""You are the Omni-OS Component Extraction Engine.
Your task is to map all interactive and highly relevant dynamic elements inside a specific UI container.

CONTAINER CONTEXT:
- ID: '{container_id}'
- Description: '{container_description}'
{padding_notice}
{padding_notice}
RULES:
1. Coordinate System: Imagine the ENTIRE provided image is exactly 1000x1000 units. Output coordinates using this 0-1000 scale relative to the image boundaries.
2. DYNAMIC COMPONENT STRUCTURING:
Map the UI using standalone elements, a repeating template, or BOTH, depending on the visual layout:
- STANDALONE ELEMENTS: Extract unique, non-repeating interactive components (e.g., individual buttons, headers, search bars, toggles) directly into the root `elements` array.
- REPEATING COLLECTIONS (Lists & Grids): 
 a) Define ONE representative item's internal structure in `item_template`. Coordinates inside `item_template.elements` must be relative to the ITEM'S bounding box (0-1000 scale).
 b) Identify the bounding boxes for every visible item in the collection and place them in the `items` array.
3. CRITICAL BOUNDING BOX GEOMETRY:
Bounding boxes (`xmin`, `xmax`, `ymin`, `ymax`) MUST encapsulate the ENTIRE perimeter of the item, not just a fragment of it.
- VISUAL BOUNDARY (Primary Rule): If the items sit inside a distinct visible container (e.g., a card with a different background color, a drawn border, or a dividing line), the bounding box MUST be the exact outer borders of that colored background. Encompass the entire visual card or row.
- LOGICAL GROUPING (Fallback): If there is no distinct background color or border, the bounding box must span the full physical height and width of the grouped data. It must encapsulate the primary content (whether that is an image, a large number, or a text block) AND all associated metadata (titles, subtitles, timestamps, action buttons) belonging to that specific item. Do not shrink or truncate the box to exclude text.
4. TEXT ANCHORING RULE (Applies ONLY to `center_x_1000` / `center_y_1000` of internal elements, NOT bounding boxes):
Text string lengths vary. Never place click coordinates in empty space:
- For LEFT-ALIGNED text: Anchor center_x_1000 near the START of the text slot (e.g., 5% to 15% past where the text begins). 
- For RIGHT-ALIGNED text: Anchor center_x_1000 near the END of the text slot (85% to 95%).
- For CENTERED text: Anchor at the horizontal midpoint (50%).
5. Do not map purely decorative backgrounds."""

        prompt = "Map the interactive components in this container crop."
        uploaded_file = self.client.files.upload(file=process_image_path)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[uploaded_file, prompt],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0, 
                    response_mime_type="application/json",
                    response_schema=ContainerExtraction
                )
            )
            
            raw_data = response.parsed

            reasoning = raw_data.thought_process
            
            final_data = {
                "elements": [],
                "item_template": None,
                "items": []
            }

            for el in raw_data.elements:
                final_data["elements"].append(
                    self._format_element(el, crop_width, crop_height, target_width, target_height)
                )

            if raw_data.item_template:
                final_data["item_template"] = {
                    "template_id": raw_data.item_template.template_id,
                    "description": raw_data.item_template.description,
                    "elements": [
                        self._format_element(tel, crop_width, crop_height, target_width, target_height, is_template_child=True) 
                        for tel in raw_data.item_template.elements
                    ]
                }

            for item in raw_data.items:
                pixel_x = (item.xmin / 1000.0) * target_width
                pixel_y = (item.ymin / 1000.0) * target_height
                pixel_x2 = (item.xmax / 1000.0) * target_width
                pixel_y2 = (item.ymax / 1000.0) * target_height
                
                x = max(0, int(pixel_x))
                y = max(0, int(pixel_y))
                w = min(crop_width, int(pixel_x2)) - x
                h = min(crop_height, int(pixel_y2)) - y
                
                final_data["items"].append({
                    "index": item.index,
                    "bbox": {"x": x, "y": y, "w": w, "h": h}
                })
                
            token_usage = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count
            }

            return reasoning, final_data, token_usage

        finally:
            self.client.files.delete(name=uploaded_file.name)
            if needs_padding and os.path.exists(process_image_path):
                os.remove(process_image_path)

    def _format_element(self, llm_el: LLMElement, crop_width: int, crop_height: int, target_width: int, target_height: int, is_template_child: bool = False) -> dict:
        """Denormalizes dynamically padded coordinates back to original crop relative ratios."""
        
        if is_template_child:
            rel_x = round(llm_el.center_x_1000 / 1000.0, 4)
            rel_y = round(llm_el.center_y_1000 / 1000.0, 4)
        else:
            pixel_x = (llm_el.center_x_1000 / 1000.0) * target_width
            pixel_y = (llm_el.center_y_1000 / 1000.0) * target_height
            
            rel_x = max(0.0, min(1.0, round(pixel_x / crop_width, 4)))
            rel_y = max(0.0, min(1.0, round(pixel_y / crop_height, 4)))

        return {
            "id": llm_el.id,
            "text": llm_el.text,
            "element_type": llm_el.element_type,
            "description": llm_el.description,
            "text_align": llm_el.text_align,
            "rel_x": rel_x,
            "rel_y": rel_y,
            "is_dynamic": llm_el.is_dynamic,
            "opens_container": llm_el.opens_container,
            "navigates_to_view": llm_el.navigates_to_view
        }