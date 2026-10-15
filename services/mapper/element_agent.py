import json
import os
import tempfile
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from PIL import Image


# ==========================================
# 1. LLM-Specific Schemas (1000-Scale)
# ==========================================
class LLMElement(BaseModel):
    id: str = Field(description="Unique snake_case identifier")
    text: Optional[str] = Field(default=None, description="Visible text if static, or placeholder role if dynamic")
    element_type: Literal['button', 'icon', 'input', 'text', 'image', 'dropdown']
    description: str = Field(description="Semantic role and behavior of the element")
    text_align: Optional[Literal['left', 'center', 'right']] = Field(default=None)

    center_x_1000: int
    center_y_1000: int


class LLMTemplateElement(BaseModel):
    id: str = Field(description="Unique snake_case identifier")
    element_type: Literal['button', 'icon', 'input', 'text', 'image', 'dropdown']
    description: str = Field(description="Semantic role of the element")

    # Kept for static anchors (e.g., a static "Play" button or "Add" text)
    text: Optional[str] = Field(default=None, description="The text, ONLY if is_dynamic is False")

    # Kept as a lightweight spatial shortcut for the Planner
    center_x_1000: int = Field(description="Relative center X (0-1000) inside the idealized card")
    center_y_1000: int = Field(description="Relative center Y (0-1000) inside the idealized card")


class BBox1000(BaseModel):
    xmin: int
    ymin: int
    xmax: int
    ymax: int


class LLMTemplate(BaseModel):
    template_id: str = Field(description="snake_case identifier (e.g. 'playlist_card', 'editor_tab')")
    description: str = Field(description="Describes what an item in this collection represents and does")

    collection_bounds: BBox1000 = Field(
        description="The outer bounding box (0-1000) of the ENTIRE region where these items are displayed. Used for masking dynamic content."
    )

    elements: List[LLMTemplateElement] = Field(description="The internal interactive parts of ONE representative item")


class ContainerExtraction(BaseModel):
    thought_process: str = Field(
        description="Step 1: List all standalone interactive elements. Step 2: Identify if a repeating template exists. Step 3: If yes, describe the template's outer boundaries before outputting coordinates."
    )
    elements: List[LLMElement] = Field(default_factory=list, description="Unique, non-repeating static elements")
    item_template: Optional[LLMTemplate] = Field(default=None, description="Template definition if the container holds repeating items")


class CandidateContainerExtraction(BaseModel):
    thought_process: str = Field(
        description=(
            "Step 1: List all standalone interactive elements visible in the image. Step 2: Compare the visible elements and container purpose against KNOWN_CANDIDATES to check for an exact, subset, or superset match, and reuse the candidate's IDs for all matching elements. Step 3: Identify if a repeating template exists and describe its outer boundaries before outputting coordinates."
        )
    )
    matched_container_id: Optional[str] = Field(
        default=None,
        description="The container_id from KNOWN_CANDIDATES if this image is an exact, subset, or superset match of that container; otherwise null."
    )
    elements: List[LLMElement] = Field(default_factory=list, description="Unique, non-repeating static elements currently visible in the image")
    item_template: Optional[LLMTemplate] = Field(default=None, description="Template definition if the container holds repeating items")


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
        container_description: str,
        candidates: Optional[List[Dict[str, Any]]] = None,
        enable_candidate_matching: bool = False,
    ) -> tuple[str, dict, dict]:
        """
        Analyzes a cropped container using dynamic padding for extreme aspect ratios.

        When `enable_candidate_matching` is False or `candidates` is empty/None, the prompt and schema
        are 100% identical to the baseline zero-memory mapper.
        """
        use_candidates = bool(enable_candidate_matching and candidates)

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
This image was padded with a solid black background to preserve resolution. 
CRITICAL COORDINATE RULE: The 0-1000 coordinate grid applies to the ENTIRE image, INCLUDING the black padding. Do not restrict your 1000 scale to just the UI area. The coordinate 1000 represents the extreme bottom/right of the black void.
"""

        system_instruction = f"""You are the Omni-OS Component Extraction Engine.
Your task is to identify interactive components and repeating behavioral patterns in a UI container.

CONTAINER CONTEXT:
- ID: '{container_id}'
- Description: '{container_description}'
{padding_notice}

RULES:
1. COORDINATE SCALE: Use a 0-1000 coordinate scale relative to the image boundaries.

2. WHAT IS A "STATE"?
   A "state" is a distinct UI layout, menu, overlay, or application window. 
   - Opening different apps (Spotify vs Firefox) = Different States.
   - Opening a display settings panel vs a network settings panel = Different States.
   - Clicking different songs in a playlist (data changes, layout remains the same) = SAME State.

3. STANDALONE VS TEMPLATES (CRITICAL DEFINITION):
   Scan for BOTH standalone elements and templates. DO NOT map raw data rows (e.g., lines of code) or purely decorative backgrounds.

   - STANDALONE ELEMENTS (Structurally Distinct States): Map unique controls that lead to fundamentally different UI layouts. For standalone items, ONLY map elements that trigger a UI action or state change. DO NOT map purely informative static text (e.g., page headers like "About the artist" or "Now Playing").
     * Example NO (Don't use template): OS App Dock. (Firefox opens a browser, Spotify opens a media player. These are distinct states. Map them as standalone `icon` elements).

   - TEMPLATES (Structurally Identical States): Use `item_template` ONLY if interacting with any item in the collection leads to the exact same TYPE of UI view or menu.
     * Example YES: A row of playlist cards. (Clicking any card opens a "Playlist View").
     * - Example YES: VS Code editor tabs. (Clicking any tab opens an "Editor View").

4. REPEATING ITEM TEMPLATES (If Rule 3 qualifies as YES):
   a) Define `collection_bounds`: Draw a bounding box (`xmin`, `xmax`, `ymin`, `ymax` in 0-1000 scale) around the ENTIRE viewport region containing the collection. Capture the FULL interactive footprint. For borderless tabs or lists, extend bounds to the natural UI boundaries (nearest divider line, container edges).
   b) Define ONE representative item in `item_template`.
   c) Map its internal parts in `item_template.elements`. UNLIKE standalone elements, inside a template YOU MUST INCLUDE both the actionable buttons AND the primary identifying visual/text anchors (e.g., thumbnail, title). These anchors are required so the system can distinguish between items, even if they do not trigger a separate action from the card itself. Coordinates must be relative to a SINGLE idealized item's bounds (0-1000 scale).

5. COORDINATE ACCURACY & ANCHORING:
   How you calculate `center_x_1000` and `center_y_1000` depends strictly on whether the element is standalone or inside a template:
   
   - FOR STANDALONE ELEMENTS (`elements` array):
     Target the EXACT physical mathematical center of the element. For buttons, icons, and avatars, the coordinate must land dead-center inside the visible shape. Do not apply offset anchoring.
     
   - FOR TEMPLATE ELEMENTS (`item_template.elements`):
     Ignore variations in text length or slight width differences across real items. Think in terms of layout anchors relative to the idealized item bounds:
     * Icons/Buttons: Pin them where they are structurally anchored (e.g., "vertically centered on the far left").
     * Text strings: DO NOT aim for the mathematical center, as text length varies. Anchor `center_x_1000` exactly where the text block BEGINS plus a small safeguard margin (e.g., 10-15% past the left edge)."""

        if use_candidates:
            sanitized_candidates = self._serialize_candidates_for_prompt(candidates)
            system_instruction += f"""

6. CANDIDATE MATCHING & SUBSET/SUPERSET ID REUSE:
   Below is a list of `KNOWN_CANDIDATES` showing previously mapped containers and their elements (`id`, `element_type`, `text`, `description`).
   a) WHEN TO MATCH (`matched_container_id`):
      Set `matched_container_id` to a candidate's `container_id` if the image represents the same functional UI container, including when:
      - EXACT MATCH: The visible elements in the image are the same as the candidate's elements.
      - SUBSET MATCH: The visible elements in the image are a subset of the candidate's elements (some candidate elements are currently hidden or collapsed).
      - SUPERSET MATCH: The candidate's elements are a subset of the visible elements in the image (new buttons, tabs, or controls are now visible alongside the candidate's elements).
      If none of the candidates share the container's functional identity and core elements, set `matched_container_id` to null.
   b) ELEMENT ID REUSE & NEW ELEMENTS:
      - For every visible element or template in the image that corresponds to an element in the matched candidate, you MUST reuse its exact `id` (and `template_id`).
      - If the image is a SUPERSET (contains additional visible interactive elements not listed in the candidate), map those new elements as well and assign them new descriptive snake_case `id`s.
      - If the image is a SUBSET (some elements from the candidate are not visible in the image), DO NOT include the missing elements—only output elements actually visible in the image.

KNOWN_CANDIDATES:
{json.dumps(sanitized_candidates, indent=2, ensure_ascii=False)}"""

        prompt = "Map the interactive components in this container crop."
        schema_cls = CandidateContainerExtraction if use_candidates else ContainerExtraction
        uploaded_file = self.client.files.upload(file=process_image_path)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[uploaded_file, prompt],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=schema_cls
                )
            )

            raw_data = response.parsed
            reasoning = raw_data.thought_process
            matched_id = getattr(raw_data, "matched_container_id", None) if use_candidates else None

            # Lookup table to preserve previously learned `opens_container` list from the matched candidate
            prior_opens: Dict[str, Optional[List[str]]] = {}
            if matched_id and candidates:
                for cand in candidates:
                    if cand.get("container_id") == matched_id:
                        for pel in cand.get("elements", []):
                            if pel.get("id") and pel.get("opens_container"):
                                prior_opens[pel["id"]] = list(pel["opens_container"])
                        break

            final_data: Dict[str, Any] = {
                "elements": [],
                "item_template": None,
            }
            if use_candidates:
                final_data["matched_container_id"] = matched_id

            for el in raw_data.elements:
                formatted_el = self._format_static_element(
                    el,
                    crop_width,
                    crop_height,
                    target_width,
                    target_height,
                    opens_container=prior_opens.get(el.id),
                )
                final_data["elements"].append(formatted_el)

            if raw_data.item_template:
                # Denormalize the global collection mask back to original crop pixels
                cb = raw_data.item_template.collection_bounds
                pixel_x = (cb.xmin / 1000.0) * target_width
                pixel_y = (cb.ymin / 1000.0) * target_height
                pixel_x2 = (cb.xmax / 1000.0) * target_width
                pixel_y2 = (cb.ymax / 1000.0) * target_height

                # Clamp to remove any padding
                x_abs = max(0, int(pixel_x))
                y_abs = max(0, int(pixel_y))
                w_abs = min(crop_width, int(pixel_x2)) - x_abs
                h_abs = min(crop_height, int(pixel_y2)) - y_abs

                # Convert to Relative Coordinates (0.0 to 1.0) relative to the container crop
                final_data["item_template"] = {
                    "template_id": raw_data.item_template.template_id,
                    "description": raw_data.item_template.description,
                    "collection_bounds": {
                        "x": round(x_abs / crop_width, 4),
                        "y": round(y_abs / crop_height, 4),
                        "w": round(w_abs / crop_width, 4),
                        "h": round(h_abs / crop_height, 4)
                    },
                    "elements": [
                        self._format_template_element(tel)
                        for tel in raw_data.item_template.elements
                    ]
                }

            token_usage = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count
            }

            return reasoning, final_data, token_usage

        finally:
            self.client.files.delete(name=uploaded_file.name)
            if needs_padding and os.path.exists(process_image_path):
                os.remove(process_image_path)

    @staticmethod
    def _serialize_candidates_for_prompt(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Serializes only semantic fields (id, type, text, description) without any coordinate data."""
        cleaned = []
        for c in candidates:
            entry: Dict[str, Any] = {
                "container_id": c.get("container_id") or c.get("id"),
                "description": c.get("description", ""),
                "elements": [
                    {
                        "id": el.get("id"),
                        "element_type": el.get("element_type"),
                        "text": el.get("text"),
                        "description": el.get("description"),
                    }
                    for el in c.get("elements", [])
                ],
            }
            if c.get("item_template"):
                tmpl = c["item_template"]
                entry["item_template"] = {
                    "template_id": tmpl.get("template_id"),
                    "description": tmpl.get("description"),
                    "elements": [
                        {
                            "id": tel.get("id"),
                            "element_type": tel.get("element_type"),
                            "text": tel.get("text"),
                            "description": tel.get("description"),
                        }
                        for tel in tmpl.get("elements", [])
                    ],
                }
            cleaned.append(entry)
        return cleaned

    def _format_static_element(
        self,
        llm_el: LLMElement,
        crop_width: int,
        crop_height: int,
        target_width: int,
        target_height: int,
        opens_container: Optional[List[str]] = None,
    ) -> dict:
        """Denormalizes dynamically padded coordinates back to original crop relative ratios for static elements."""
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
            "opens_container": opens_container,
        }

    def _format_template_element(self, llm_tel: LLMTemplateElement) -> dict:
        """Processes template elements, keeping coordinates strictly internal/relative."""
        return {
            "id": llm_tel.id,
            "text": llm_tel.text,
            "element_type": llm_tel.element_type,
            "description": llm_tel.description,
            "rel_x": round(llm_tel.center_x_1000 / 1000.0, 4),
            "rel_y": round(llm_tel.center_y_1000 / 1000.0, 4)
        }