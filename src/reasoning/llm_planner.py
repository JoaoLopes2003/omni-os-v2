import os
import textwrap
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Literal
from PIL import Image
from src.core.schemas import UIElement

class ActionPlan(BaseModel):
    reasoning: str = Field(
        description="Step-by-step spatial reasoning explaining the visual cues and decisions."
    )
    action: Literal["click", "type", "done", "click_point", "wait", "abort"]
    element_id: int = Field(
        default=0,
        description="The numeric Set-of-Marks ID. Must be set to 0 if using type, click_point, wait, done, or abort."
    )
    text_input: str = Field(
        default="",
        description="The text to type if action is 'type'. Otherwise leave empty."
    )
    point_x: int = Field(
        default=-1,
        description="X coordinate on a 0-1000 normalized grid. MANDATORY if action is 'click_point', otherwise -1."
    )
    point_y: int = Field(
        default=-1,
        description="Y coordinate on a 0-1000 normalized grid. MANDATORY if action is 'click_point', otherwise -1."
    )

class OSPlannerEngine:
    def __init__(self):
        """
        Initializes the Gemini API client.
        """
        key = os.getenv('GEMINI_API_KEY')
        if not key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        self.client = genai.Client(api_key=key)

    def plan_action(self, image_path: str, elements: dict[int, UIElement], user_prompt: str, history: list[str]) -> tuple[ActionPlan, str]:
        """
        Analyzes the grounded UI and returns a JSON action plan.
        """
        # Load the image using PIL (The SDK handles the bytes automatically)
        img = Image.open(image_path)
        
        # Build the UI context string
        # We give the LLM a text cheat-sheet of the elements to pair with the image
        context_str = "Available UI Elements:\n"
        for id, el in elements.items():
            context_str += f"ID: {id} | Text: '{el.text}'\n"

        history_str = "No previous actions taken." if not history else "\n".join(history)
        
        # Master Prompt
        prompt = f"""You are an autonomous Operating System GUI Agent.
Your goal is to help the user navigate the screen based on their request.

I have provided you with a screenshot of the current UI. The image has been processed with a Set-of-Marks algorithm: interactable elements are highlighted with green bounding boxes and labeled with a numeric ID in red (e.g., [ 1 ]).
I have also provided a text list mapping these IDs to their extracted text.

USER REQUEST: "{user_prompt}"

PAST ACTIONS (Chronological Order):
{history_str}

{context_str}

ANTI-LOOP & STATE PROTOCOLS:
- If the screen is clearly loading (e.g., blank windows, loading spinners), you must wait.
- Review the PAST ACTIONS. If you have attempted the exact same action 2 times without achieving the desired state, you are stuck in a loop. You MUST try a different element, or use the "abort" action.

INSTRUCTIONS:
1. Read the PAST ACTIONS and apply the ANTI-LOOP PROTOCOL.
2. Analyze the user request.
3. Look at the provided screenshot to locate the UI element that fulfills the request.
4. Cross-reference the visual ID with the text list provided.
5. Provide your step-by-step spatial reasoning in the "reasoning" field.
6. Choose exactly one "action" from this list:
   - "click": (PRIMARY) Use this to click buttons, links, or to focus input fields that HAVE a red ID tag. Provide the "element_id".
   - "click_point": (FALLBACK) Use this ONLY to click icons or images that DO NOT have a red ID tag. You MUST set "element_id" to 0 and provide the exact integer values for "point_x" and "point_y" (0 to 1000 scale).
   - "type": Use this ONLY to type text on your keyboard. It assumes the input field is ALREADY focused from a previous click. You MUST set "element_id" to 0. Put the text in the "text_input" field.
   - "wait": Use this if an application or page is loading and you need more time.
   - "abort": Use this if you are stuck in an infinite loop, or if the user's request is impossible to fulfill.
   - "done": Use this when the overall USER REQUEST has been completely fulfilled.
7. CRITICAL RULE FOR CLICK_POINT: If "action" is "click_point", "point_x" and "point_y" CANNOT be -1. You must estimate the pixel coordinate on a 1000x1000 grid (e.g., top-left is [0, 0], center is [500, 500], bottom-right is [1000, 1000]).
8. Select the exact "element_id" to interact with. If no element matches, set element_id to 0.
9. Output strict JSON matching the requested schema. Do not include markdown formatting like ```json."""
        
        # Call the standard models API
        response = self.client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ActionPlan,
                temperature=0.0  # Force deterministic output
            )
        )
        
        # Validate the response text against our Pydantic model
        result = ActionPlan.model_validate_json(response.text)

        return result, prompt
        

if __name__ == "__main__":
    # --- Dummy Test Block ---
    from dotenv import load_dotenv
    load_dotenv()
    
    print("Instantiating Planner Engine...")
    planner = OSPlannerEngine()
    
    # We will create a dummy element simulating the "Firefox" icon from your previous screenshot
    dummy_elements = {
        8: UIElement(
            element_id=8, 
            bbox={"xmin": 0.1, "ymin": 0.1, "xmax": 0.2, "ymax": 0.2}, # Dummy normalized bbox
            text="Firefox", 
            confidence=0.99
        )
    }
    
    # Assuming "final_desktop_som.png" is in the root directory from the previous step
    image_file = "final_desktop_som.png"
    user_instruction = "Click on the Firefox icon."
    
    if os.path.exists(image_file):
        print(f"Requesting plan for: '{user_instruction}'...")
        plan = planner.plan_action(image_file, dummy_elements, user_instruction)
        
        print("\n=== LLM ACTION PLAN ===")
        print(f"Reasoning: {plan.reasoning}")
        print(f"Action: {plan.action}")
        print(f"Target ID: {plan.element_id}")
    else:
        print(f"Error: Could not find {image_file}. Make sure you run main.py first to generate the image.")