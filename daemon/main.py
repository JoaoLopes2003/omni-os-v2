import time
import os
import mss
import requests
import pywinctl
import pyautogui
import pyperclip
import mss.tools
import json
import subprocess
import argparse

from daemon.core.window import UniversalWindowObserver
from daemon.core.router import StateRouter
from shared.core.graph_db import GraphDatabase
from daemon.logger import ExecutionLogger
from daemon.metrics import MetricsTracker

class OmniOSDaemon:
    def __init__(self, debug_mode: bool = False):
        print(f"[Daemon] Booting Omni-OS v2 Core Systems... (Debug: {debug_mode})")
        self.debug_mode = debug_mode
        self.user_profile_text = "No specific preferences provided."
        self.observer = UniversalWindowObserver()
        self.router = StateRouter()
        self.db = GraphDatabase()

        # Initialize Logger and Metrics Tracker
        self.logger = ExecutionLogger(debug_mode=self.debug_mode)
        self.metrics = MetricsTracker(output_dir=self.logger.session_dir, debug_mode=self.debug_mode)

    def _send_to_mapper(self, state_id: str, window_title: str, image_path: str):
        """Helper function to keep the network logic DRY."""
        try:
            with open(image_path, "rb") as img_file:
                start_time = time.time()
                response = requests.post(
                    "http://localhost:8001/map",
                    data={
                        "state_id": state_id, 
                        "window_title": window_title,
                        "debug": str(self.debug_mode).lower()  # Send the flag
                    },
                    files={"image": img_file}
                )
                duration = time.time() - start_time
            if response.status_code == 200:
                res_data = response.json()
                print(f"[Daemon] Mapper Success: {res_data['message']}")
                
                # Track metrics if returned by the Mapper endpoint
                metrics_data = res_data.get("metrics", {})
                if metrics_data:
                    self.metrics.log_cache_event(hit=False) # Mapper execution means cache miss
                    self.metrics.log_system_op("OCR_Extraction", metrics_data.get("ocr_time", 0.0))
                    
                    tokens = metrics_data.get("token_usage", {"input_tokens": 0, "output_tokens": 0})
                    self.metrics.log_model_call(
                        agent="Mapper",
                        model_name="gemini-3.1-pro-preview",
                        duration=metrics_data.get("llm_time", duration),
                        input_tokens=tokens["input_tokens"],
                        output_tokens=tokens["output_tokens"]
                    )
            else:
                print(f"[Daemon] Mapper Failed: {response.text}")
        except requests.exceptions.ConnectionError:
            print("[Daemon] Error: Could not connect to Mapper on port 8001.")

    def _load_user_profile(self):
        profile_path = "shared/config/user_profile.txt"
        if os.path.exists(profile_path):
            with open(profile_path, "r", encoding="utf-8") as f:
                self.user_profile_text = f.read()

    def _bootstrap_system(self):
        """Forces a clean mapping of the OS desktop and pre-warms target apps. Also loads the user preferences."""
        self._load_user_profile()
        
        # Mandatory Root Desktop Mapping
        state_id = "os_desktop"
        
        if not self.db.get_state(state_id):
            print(f"\n[Daemon] Bootstrapping root state: '{state_id}'...")
            pyautogui.hotkey('win', 'd')
            time.sleep(1.5)
            
            screenshot_path = "tmp/temp_desktop_capture.png"
            with mss.MSS() as sct:
                sct.shot(mon=1, output=screenshot_path)
                
            self._send_to_mapper(state_id, "Desktop", screenshot_path)
            
            if os.path.exists(screenshot_path):
                os.remove(screenshot_path)
                
            pyautogui.hotkey('win', 'd')
            time.sleep(1)
            print("[Daemon] Root state bootstrap complete.")
        else:
            print(f"[Daemon] Root state '{state_id}' is already cached.")

        # Application Pre-warming
        prewarm_file = "shared/config/prewarm.json"
        if not os.path.exists(prewarm_file):
            return

        with open(prewarm_file, "r") as f:
            targets = json.load(f).get("prewarm_targets", [])

        if not targets:
            return

        print("\n[Daemon] Initiating Cache Pre-warming...")

        for target in targets:
            target_cmd = target.get("cmd")
            expected_state = target.get("expected_state")
            
            # Use the Router to confidently find the right window
            all_windows = pywinctl.getAllWindows()
            matching_window = None
            for w in all_windows:
                if self.router.resolve_state_id(w.title) == expected_state:
                    matching_window = w
                    break
            
            # Smart Launch & Polling
            if not matching_window:
                print(f"[Daemon] Target for '{expected_state}' is closed. Launching via '{target_cmd}'...")
                subprocess.Popen(target_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                timeout = 15
                start_time = time.time()
                while time.time() - start_time < timeout:
                    all_windows = pywinctl.getAllWindows()
                    for w in all_windows:
                        if self.router.resolve_state_id(w.title) == expected_state:
                            matching_window = w
                            break
                            
                    if matching_window:
                        print(f"[Daemon] Window detected! Waiting 3s for UI paint...")
                        time.sleep(3)
                        break
                    time.sleep(0.5)
                    
                if not matching_window:
                    print(f"[Daemon] Timeout waiting for '{expected_state}' to open. Skipping.")
                    print(f"[DEBUG] Window titles currently visible to pywinctl:")
                    for w in all_windows:
                        if w.title.strip():  # Only print non-empty titles
                            print(f"  -> '{w.title}'")
                    continue

            # We know the target_state_id is exactly the expected_state
            if self.db.get_state(expected_state):
                print(f"[Daemon] Skipping '{expected_state}': Already cached.")
                continue

            print(f"[Daemon] Pre-warming '{expected_state}'...")
            matching_window.activate()
            time.sleep(1) 

            is_maximized = matching_window.isMaximized
            if not is_maximized:
                matching_window.maximize()
                time.sleep(1.5)

            monitor_bbox = {
                "top": max(0, matching_window.top),
                "left": max(0, matching_window.left),
                "width": matching_window.width,
                "height": matching_window.height
            }

            screenshot_path = "tmp/temp_capture.png"
            with mss.MSS() as sct:
                img = sct.grab(monitor_bbox)
                mss.tools.to_png(img.rgb, img.size, output=screenshot_path)

            self._send_to_mapper(expected_state, matching_window.title, screenshot_path)

            if os.path.exists(screenshot_path):
                os.remove(screenshot_path)
                
            if not is_maximized:
                matching_window.restore()

        print("[Daemon] Cache Pre-warming complete.\n")

    def _execute_action(self, action: dict, active_window, state_graph):
        """Translates a JSON action into physical OS operations."""
        a_type = action.get("action_type")
        history_entry = f"Executed {a_type}"

        if a_type == "click_element":
            target_id = action.get("target_id")
            # Find the element in the graph
            element = next((e for e in state_graph.elements if e.id == target_id), None)
            if element:
                # Denormalize coordinates relative to the active window
                x = int(active_window.left + (element.center_x / 1000 * active_window.width))
                y = int(active_window.top + (element.center_y / 1000 * active_window.height))
                pyautogui.click(x, y, duration=0.5, button="left")
                history_entry += f" on '{target_id}' at ({x}, {y})"
            else:
                print(f"[Execution Error] Element '{target_id}' not found in state graph.")
                history_entry += f" (FAILED: '{target_id}' not found)"

        elif a_type == "click_coordinate":
            tx = action.get("target_x", 500)
            ty = action.get("target_y", 500)
            x = int(active_window.left + (tx / 1000 * active_window.width))
            y = int(active_window.top + (ty / 1000 * active_window.height))
            pyautogui.click(x, y, duration=0.5, button="left")
            history_entry += f" at relative ({tx}, {ty})"

        elif a_type == "type_text":
            payload = action.get("text_payload", "")
            pyperclip.copy(payload)
            time.sleep(0.1)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.1)
            pyperclip.copy("")
            history_entry += f": '{payload}'"

        elif a_type == "hotkey":
            combo = action.get("hotkey_combo", [])
            pyautogui.hotkey(*combo)
            history_entry += f" {combo}"

        elif a_type == "run_cli_command":
            cmd = action.get("text_payload", "")
            try:
                # Capture output, timeout after 10s to prevent hanging
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
                
                # Check the exit code (0 means success, anything else is an error)
                if result.returncode == 0:
                    if result.stdout.strip():
                        # Success with output
                        output_str = result.stdout.strip()
                        truncated = output_str[:1000] + ("..." if len(output_str) > 1000 else "")
                        self.memory["temp_cli_output"] = truncated
                        history_entry += f": '{cmd}' (Success. Output saved to 'temp_cli_output')"
                    else:
                        # Success but silent (e.g., mkdir, rm)
                        self.memory["temp_cli_output"] = "Command executed successfully (No output)."
                        history_entry += f": '{cmd}' (Success: No output)"
                else:
                    # The command failed
                    error_str = result.stderr.strip() or result.stdout.strip() or "Unknown error."
                    truncated_error = error_str[:1000] + ("..." if len(error_str) > 1000 else "")
                    self.memory["temp_cli_output"] = f"ERROR (Exit {result.returncode}): {truncated_error}"
                    history_entry += f": '{cmd}' (FAILED. Error saved to 'temp_cli_output')"

            except subprocess.TimeoutExpired:
                self.memory["temp_cli_output"] = "Command timed out after 10 seconds."
                history_entry += f": '{cmd}' (FAILED: Timeout)"

        elif a_type == "save_clipboard_to_memory":
            label = action.get("memory_label", "clipboard_var")
            # Wait a tiny bit just to ensure OS clipboard has updated from a previous click
            time.sleep(0.2) 
            val = pyperclip.paste()
            self.memory[label] = val
            history_entry += f" (Saved clipboard to '{label}': '{val[:30]}...')"

        elif a_type == "extract_data":
            label = action.get("memory_label", "unknown_var")
            val = action.get("text_payload", "")
            self.memory[label] = val
            history_entry += f" ({label} = '{val}')"

        elif a_type == "switch_window":
            target_title = action.get("text_payload", "")
            windows = pywinctl.getAllWindows()
            target_win = next((w for w in windows if target_title.lower() in w.title.lower()), None)
            if target_win:
                target_win.activate()
                time.sleep(0.5)
                history_entry += f" to '{target_title}'"
            else:
                history_entry += f" (FAILED: window '{target_title}' not found)"

        elif a_type == "wait":
            wait_time = action.get("wait_seconds", 2)
            time.sleep(wait_time)
            history_entry += f" for {wait_time}s"

        return history_entry
        
    def run(self, user_goal: str):
        self._bootstrap_system()

        self.logger.start_user_prompt(user_goal)

        print(f"\n[Daemon] Task Accepted: '{user_goal}'")
        print("[Daemon] Consulting Master Planner...")
        
        # Ask the Master Planner to decompose the goal
        try:
            start_time = time.time()
            response = requests.post(
                "http://localhost:8002/decompose",
                data={
                    "user_goal": user_goal,
                    "user_profile": self.user_profile_text
                }
            )
            duration = time.time() - start_time

            if response.status_code == 200:
                decomposition = response.json()

                tokens = decomposition.get("token_usage", {"input_tokens": 0, "output_tokens": 0})
                self.metrics.log_model_call(
                    agent="Master",
                    model_name="gemini-3.8-flash",
                    duration=duration,
                    input_tokens=tokens["input_tokens"],
                    output_tokens=tokens["output_tokens"]
                )

                self.logger.log_master_planner(
                    raw_prompt=decomposition.get("raw_prompt", ""), 
                    response=decomposition
                )

                subgoals = decomposition.get("subgoals", [])
                
                # Initialize memory with the Master Planner's extracted variables
                raw_memory = decomposition.get("initial_memory", [])
                self.memory = {item["key"]: item["value"] for item in raw_memory}
                
                print(f"[Daemon] Thought Process: {decomposition.get('thought_process')}")
                print(f"[Daemon] Extracted {len(self.memory)} initial variables.")
                print(f"[Daemon] Generated {len(subgoals)} subgoals.")
            else:
                print(f"[Daemon] Master Planner API Failed: {response.text}")
                return
        except requests.exceptions.ConnectionError:
            print("[Daemon] Error: Could not connect to Master Planner on port 8002.")
            return

        print("\n[Daemon] Entering autonomous execution. Press Ctrl+C to abort.")
        
        # Iterate through Subgoals (The Outer Loop)
        for i, subgoal in enumerate(subgoals):
            current_subgoal_text = subgoal.get("description")

            self.logger.start_subgoal(i + 1, current_subgoal_text)

            print(f"\n==========================================")
            print(f"[Daemon] Executing Subgoal {i + 1}/{len(subgoals)}")
            print(f"[Daemon] Target: {current_subgoal_text}")
            print(f"==========================================")
            
            # Reset action history for each new subgoal to prevent context bleeding
            self.action_history = []
            step_count = 1
            subgoal_complete = False
            
            # Action Planner Execution (The Inner Loop)
            try:
                while not subgoal_complete:
                    print(f"\n--- Step {step_count} ---")
                    
                    # Sense the Environment
                    active_window = pywinctl.getActiveWindow()
                    window_title = active_window.title if active_window else "Desktop"
                    state_id = self.router.resolve_state_id(window_title)
                    
                    print(f"[Daemon] Active Window: '{window_title}'")
                    print(f"[Daemon] Resolved State: '{state_id}'")
                    
                    if active_window:
                        monitor_bbox = {
                            "top": max(0, active_window.top),
                            "left": max(0, active_window.left),
                            "width": active_window.width,
                            "height": active_window.height
                        }
                    else:
                        monitor_bbox = 1 

                    screenshot_path = "tmp/temp_capture.png"
                    with mss.MSS() as sct:
                        img = sct.grab(monitor_bbox)
                        mss.tools.to_png(img.rgb, img.size, output=screenshot_path)
                    
                    state_graph = self.db.get_state(state_id)
                    
                    if state_graph:
                        print(f"[Daemon] -> CACHE HIT! Triggering Action Planner...")
                        open_windows = [w.title for w in pywinctl.getAllWindows() if w.title.strip()]
                        
                        try:
                            with open(screenshot_path, "rb") as img_file:
                                response = requests.post(
                                    "http://localhost:8003/plan",
                                    data={
                                        "user_goal": current_subgoal_text,
                                        "state_id": state_id,
                                        "state_graph": state_graph.model_dump_json(),
                                        "open_windows": json.dumps(open_windows),
                                        "action_history": json.dumps(self.action_history[-3:]), 
                                        "memory": json.dumps(self.memory) # Memory persists across all subgoals
                                    },
                                    files={"image": img_file}
                                )
                            
                            if response.status_code == 200:
                                plan_data = response.json()

                                tokens = plan_data.get("token_usage", {"input_tokens": 0, "output_tokens": 0})
                                self.metrics.log_model_call(
                                    agent="Planner",
                                    model_name="gemini-3.8-flash",
                                    duration=duration,
                                    input_tokens=tokens["input_tokens"],
                                    output_tokens=tokens["output_tokens"]
                                )

                                current_thought = plan_data.get('thought_process', '')

                                internal_state = {
                                    "active_window": window_title,
                                    "resolved_state_id": state_id,
                                    "open_windows": open_windows,
                                    "memory_dictionary": self.memory,
                                    "action_history": self.action_history
                                }
                                
                                self.logger.log_action_step(
                                    inner_step=step_count,
                                    screenshot_path=screenshot_path,
                                    raw_prompt=plan_data.get("raw_prompt", ""),
                                    response=plan_data,
                                    state=internal_state,
                                    state_graph_dict=state_graph.model_dump()
                                )

                                print(f"[Planner] Thought: {current_thought}")
                                
                                actions = plan_data.get("data", [])
                                step_action_logs = []

                                for action in actions:
                                    a_type = action.get("action_type")
                                    
                                    if a_type == "done":
                                        print(f"\n[Daemon] Subgoal {i + 1} achieved!")

                                        # Purge temporary variables before the next subgoal
                                        self.memory = {k: v for k, v in self.memory.items() if not k.startswith("temp_")}
                                        print(f"[Daemon] Memory cleaned. Retained {len(self.memory)} global variables.")

                                        subgoal_complete = True
                                        break # Breaks action loop, proceeds to next subgoal
                                        
                                    elif a_type == "abort":
                                        print("\n[Daemon] Action Planner aborted the task. Halting entire sequence.")
                                        return
                                    
                                    act_start = time.time()
                                    history_log = self._execute_action(action, active_window, state_graph)
                                    act_duration = time.time() - act_start

                                    self.metrics.log_action(a_type, act_duration)

                                    print(f"  -> {history_log}")
                                    step_action_logs.append(history_log)
                                    time.sleep(0.5)

                                # Once the loop finishes (or breaks), compile the turn and append to history
                                if step_action_logs:
                                    actions_formatted = "\n".join([f"  - {log}" for log in step_action_logs])
                                    combined_log = f"Thought: {current_thought}\nActions Executed:\n{actions_formatted}"
                                    self.action_history.append(combined_log)
                            else:
                                print(f"[Daemon] Planner API Failed: {response.text}")
                                
                        except requests.exceptions.ConnectionError:
                            print("[Daemon] Error: Could not connect to Planner.")
                            
                    else:
                        print(f"[Daemon] -> CACHE MISS! Triggering Mapper microservice...")
                        self._send_to_mapper(state_id, window_title, screenshot_path)
                        time.sleep(1)
                    
                    if os.path.exists(screenshot_path):
                        os.remove(screenshot_path)
                    
                    step_count += 1
                        
            except KeyboardInterrupt:
                print("\n[Daemon] Execution aborted by user.")
                return # Exits the entire run method
                
        print("\n[Daemon] All subgoals completed successfully. Omni-OS returning to standby.")

        self.metrics.generate_report()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Omni-OS Daemon Core")
    parser.add_argument("--debug", action="store_true", help="Enable visual debugging artifacts in shared/debug/")
    args = parser.parse_args()

    daemon = OmniOSDaemon(debug_mode=args.debug)
    daemon.run(user_goal="Go to Spotify, chose a song that my father might like (he was born in 1974) and send an email to him with the spotify link. He's email is sergio.h.s.lopes@gmail.com.")