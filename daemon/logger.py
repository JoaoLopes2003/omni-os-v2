import os
import json
import time
import shutil
import cv2

class ExecutionLogger:
    def __init__(self, base_dir="shared/debug/sessions", debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.base_dir = base_dir
        self.session_dir = os.path.join(self.base_dir, f"session_{int(time.time())}")
        
        # Only create root directories if debugging is enabled
        if self.debug_mode:
            os.makedirs(self.session_dir, exist_ok=True)
        
        self.prompt_count = 0
        self.current_prompt_dir = ""
        self.global_step_count = 0
        self.current_subgoal_dir = ""

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.thickness = 1

    def _draw_gemini_elements(self, image_path: str, elements: list, output_path: str):
        """Draws red dots and labels over the UI elements based on the state graph."""
        frame = cv2.imread(image_path)
        if frame is None:
            return
            
        h, w, _ = frame.shape
        
        for element in elements:
            cx_norm = element.get("center_x", -1)
            cy_norm = element.get("center_y", -1)
            el_type = element.get("type", element.get("element_type", "unknown"))
            label = element.get("id", element.get("text", "unnamed"))
            
            if cx_norm != -1 and cy_norm != -1:
                cx = int((cx_norm / 1000) * w)
                cy = int((cy_norm / 1000) * h)
                
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                display_text = f"{label} ({el_type})"
                cv2.putText(frame, display_text, (cx + 8, cy + 4), self.font, self.font_scale, (255, 0, 0), self.thickness)
                
        cv2.imwrite(output_path, frame)

    def start_user_prompt(self, user_goal: str):
        """Creates the root directory for a new user prompt."""
        if not self.debug_mode: return
        self.prompt_count += 1
        self.global_step_count = 1  # Reset global steps for the new prompt
        
        # Sanitize prompt for folder name
        safe_goal = "".join([c if c.isalnum() else "_" for c in user_goal[:30]])
        dir_name = f"{self.prompt_count}_Prompt_{safe_goal}"
        
        self.current_prompt_dir = os.path.join(self.session_dir, dir_name)
        os.makedirs(self.current_prompt_dir, exist_ok=True)

    def log_master_planner(self, raw_prompt: str, response: dict):
        """Logs the Master Planner's decomposition."""
        if not self.debug_mode: return
        dir_name = f"{self.global_step_count}_Master_Planner"
        target_dir = os.path.join(self.current_prompt_dir, dir_name)
        os.makedirs(target_dir, exist_ok=True)

        # Write exactly what the model read
        with open(os.path.join(target_dir, "1_raw_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(raw_prompt)
            
        with open(os.path.join(target_dir, "2_response.json"), "w", encoding="utf-8") as f:
            json.dump(response, f, indent=2)
            
        self.global_step_count += 1

    def start_subgoal(self, subgoal_index: int, description: str):
        """Creates a directory for a specific subgoal execution."""
        if not self.debug_mode: return
        dir_name = f"{self.global_step_count}_Subgoal_{subgoal_index}"
        self.current_subgoal_dir = os.path.join(self.current_prompt_dir, dir_name)
        os.makedirs(self.current_subgoal_dir, exist_ok=True)
        self.global_step_count += 1

    def log_action_step(self, inner_step: int, screenshot_path: str, raw_prompt: str, response: dict, state: dict, state_graph_dict: dict):
        if not self.debug_mode: return
        target_dir = os.path.join(self.current_subgoal_dir, f"{inner_step}_Action_Step")
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy clean screenshot
        clean_screenshot_path = os.path.join(target_dir, "1a_screenshot_clean.png")
        if os.path.exists(screenshot_path):
            shutil.copy2(screenshot_path, clean_screenshot_path)
            
            # Generate mapped screenshot
            mapped_screenshot_path = os.path.join(target_dir, "1b_screenshot_mapped.png")
            elements = state_graph_dict.get("elements", [])
            self._draw_gemini_elements(clean_screenshot_path, elements, mapped_screenshot_path)
            
        # Raw Prompt
        with open(os.path.join(target_dir, "2_raw_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(raw_prompt)
            
        # Model Response
        with open(os.path.join(target_dir, "3_response.json"), "w", encoding="utf-8") as f:
            json.dump(response, f, indent=2)
            
        # Internal OS State
        with open(os.path.join(target_dir, "4_internal_state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            
        # UI State Graph
        with open(os.path.join(target_dir, "5_state_graph.json"), "w", encoding="utf-8") as f:
            json.dump(state_graph_dict, f, indent=2)