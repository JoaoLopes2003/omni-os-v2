from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List
import json
from PIL import Image

# Adjust imports based on your monorepo structure
from shared.models.graph_models import MappedElement, ScreenStateNode
from shared.core.graph_db import GraphDatabase

class MapperOutput(BaseModel):
    """Temporary schema strictly for forcing the LLM to output a list of elements."""
    elements: List[MappedElement]

class MapperAgent:
    def __init__(self, api_key: str):
        # Initialize the modern Gemini Client
        self.client = genai.Client(api_key=api_key)
        self.db = GraphDatabase(db_dir="shared/data/states")

    def generate_state_map(self, image_path: str, window_title: str, state_id: str, raw_ocr_data: list[dict]) -> tuple[ScreenStateNode, str, dict[str, int]]:
        """
        Analyzes the clean UI and raw OCR data to generate a mapped ScreenStateNode.
        """
        
        system_instruction = f"""You are an expert OS UI Cartographer. Your job is to map the interactable elements of a software interface.
The current active window title is: '{window_title}'.

Your tasks:
1. Chrome vs Canvas Filtering: You must ONLY map static UI elements (The "Chrome"). 
    - DO MAP: Navigation bars, menu ribbons, permanent sidebars, toolbars, and system icons.
    - DO NOT MAP: Highly dynamic content (The "Canvas") such as file directory trees, open tabs, code/text editor contents, web page bodies, or terminal outputs. Completely ignore OCR data that falls inside these dynamic zones.
2. Data Cleaning: Merge fragmented OCR text into cohesive logical elements. 
    - Use the bounding box coordinates to determine if words are on the same line or inside the same button.
    - If confidence is low but the text visually matches a UI element in the image, correct the text and map it. Ignore pure visual artifacts.
    - For your FINAL output, calculate `center_x` and `center_y` by finding the middle of the bounding box (or the merged bounding boxes).
3. Icon Discovery: Look at the clean image and identify interactable icons in the static Chrome that do NOT have text (e.g., a gear icon, explorer icon, window controls). 
    - Assign them a descriptive string `id` (e.g., "icon_settings", "icon_explorer").
    - Estimate their `center_x` and `center_y` (0-1000) using the OCR text coordinates as spatial landmarks.

I am providing you with:
1. A CLEAN, unmodified screenshot of the UI.
2. A JSON list of raw OCR text detections containing their normalized bounding boxes [xmin, ymin, xmax, ymax] (0-1000 scale) and confidence scores.

Raw OCR Data (X,Y are 0-1000 normalized):
{json.dumps(raw_ocr_data, indent=2)}"""

        # Load the image using PIL (The modern SDK handles the bytes automatically)
        img = Image.open(image_path)

        print(f"[Mapper API] Analyzing state '{state_id}' via Gemini 1.5 Pro...")
        
        # Call the standard models API with GenerateContentConfig
        response = self.client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=[system_instruction, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MapperOutput
            )
        )
        
        # Parse the guaranteed JSON output back into our temporary Pydantic model
        mapper_result = MapperOutput.model_validate_json(response.text)
        
        # ==========================================
        # DETERMINISTIC STATE ASSEMBLY
        # ==========================================
        # We use the state_id passed by the Daemon, eliminating hallucination risk.
        final_state = ScreenStateNode(
            state_id=state_id,
            elements=mapper_result.elements
        )
        
        # Persist the final, clean graph to the database
        self.db.save_state(final_state)
        print(f"[Mapper] Successfully mapped {len(final_state.elements)} elements for '{state_id}'.")

        # Extract token usage
        token_usage = {
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count
        }
        
        return final_state, token_usage