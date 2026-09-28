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
    elements: List[LLMElement] = Field(default_factory=list, description="Populate ONLY if container_type is 'standard'")
    item_template: Optional[LLMTemplate] = Field(default=None, description="Populate ONLY if container_type is a collection")
    items: List[LLMCollectionItem] = Field(default_factory=list, description="Bounding boxes of the repeated items (collections only)")

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
        container_type: str, 
        container_description: str
    ) -> tuple[dict, dict]:
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
                padded_img = Image.new("RGB", (target_width, target_height), (128, 128, 128))
                padded_img.paste(img, (0, 0))  # Anchor to top-left
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    padded_img.save(temp_file, format="PNG")
                    process_image_path = temp_file.name
            else:
                process_image_path = cropped_image_path

        padding_notice = ""
        if needs_padding:
            padding_notice = """
IMAGE PADDING NOTICE: 
This UI element had an extreme aspect ratio, so it was padded with a solid gray background to preserve resolution. 
Only map the UI elements inside the actual application crop located in the top-left corner. Ignore the gray void.
"""

        system_instruction = f"""You are the Omni-OS Component Extraction Engine.
Your task is to map all interactive and highly relevant dynamic elements inside a specific UI container.

CONTAINER CONTEXT:
- ID: '{container_id}'
- Type: '{container_type}'
- Description: '{container_description}'
{padding_notice}
RULES:
1. Coordinate System: Imagine the ENTIRE provided image is exactly 1000x1000 units. Output all coordinates using this 0-1000 scale relative to the image boundaries.
2. If Type is 'standard': Extract all buttons, icons, text inputs, and major text blocks into the `elements` array. Leave `item_template` and `items` empty.
3. If Type is 'collection_grid' or 'collection_list': 
   - Define the structure of ONE repeating card/row in `item_template`. The coordinates inside the template elements must be relative to the CARD'S bounding box (0-1000 scale).
   - Identify the bounding boxes for every visible card/row in the image and put them in the `items` array using the 0-1000 scale relative to the whole image.
   - CRITICAL BOUNDING BOX RULE: The bounding box for each item MUST fully enclose the entire visual component (the full icon, text, and padding). DO NOT draw narrow slivers.
   - Leave the root `elements` array empty.
4. Do not map purely decorative backgrounds. Map things the user can click, read, or interact with."""

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

            return final_data, token_usage

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
            "rel_x": rel_x,
            "rel_y": rel_y,
            "is_dynamic": llm_el.is_dynamic,
            "opens_container": llm_el.opens_container,
            "navigates_to_view": llm_el.navigates_to_view
        }