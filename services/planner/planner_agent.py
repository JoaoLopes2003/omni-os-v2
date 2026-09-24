import json
from google import genai
from google.genai import types
from PIL import Image
from shared.models.action_models import ActionPlan

class PlannerAgent:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-3.8-flash"

    def generate_plan(
        self, 
        image_path: str, 
        user_goal: str, 
        state_id: str, 
        state_graph_json: str, 
        open_windows: list[str],
        action_history: list[str],
        memory: dict[str, str]
    ) -> ActionPlan:
        
        system_instruction = f"""You are the Omni-OS Autonomous Execution Engine.
Your job is to achieve the USER GOAL by outputting a precise sequence of OS actions.

You are currently looking at the OS state: '{state_id}'.

I am providing you with:
1. A live screenshot of the current window.
2. A JSON map of the STATIC "Chrome" elements (buttons, menus, icons).
3. A list of all currently open window titles.
4. A chronological history of your recent actions.
5. Your internal memory dictionary containing data you have extracted.

Your rules:
- To interact with a static element, output a 'click_element' action and provide its exact `target_id`.
- To interact with dynamic content, output a 'click_coordinate' action and estimate `target_x` and `target_y` (0-1000 normalized scale).
- Use 'type_text' to enter text into active fields.
- Use 'hotkey' actions (like ['ctrl', 'c']) if you know they apply.
- To switch to a different application, output a 'switch_window' action and provide the exact window title in `text_payload`.
- To run a background bash command, use 'run_cli_command' and provide the command in `text_payload`.
- To save information, use the 'extract_data' action. You MUST provide the extracted text in `text_payload` AND a variable name in `memory_label`.
- VARIABLE NAMING RULES: 
  1. If the USER GOAL explicitly asks you to store information under a specific name (e.g., "store it as `target_email`"), you MUST use that EXACT name for your `memory_label`.
  2. If you are extracting information temporarily just to help you finish the current goal, you MUST prefix your `memory_label` with `temp_` (e.g., `temp_clipboard_link`, `temp_artist_name`).
- If the user goal is fully complete, output a 'done' action.
- Keep plans short (1 to 3 actions max).

ANTI-LOOP & STATE PROTOCOLS:
- If the screen is clearly loading, output a 'wait' action.
- Review the PAST ACTIONS. If you have attempted the exact same action repeatedly without achieving the desired state, you are stuck. You MUST try a different approach or output an 'abort' action.
- If the user's request is impossible to fulfill, output an 'abort' action.

Currently Open Windows:
{json.dumps(open_windows, indent=2)}

Static Element Map:
{state_graph_json}

PAST ACTIONS (Chronological Order):
{json.dumps(action_history, indent=2)}

EXTRACTED MEMORY (Key-Value Pairs):
{json.dumps(memory, indent=2)}
"""

        img = Image.open(image_path)
        print(f"[Planner] Formulating plan for goal: '{user_goal}'...")

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[system_instruction, f"USER GOAL: {user_goal}", img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ActionPlan,
                temperature=0.0 
            )
        )
        
        plan = ActionPlan.model_validate_json(response.text)
        print(f"[Planner] Plan generated: {len(plan.actions)} actions.")
        return plan