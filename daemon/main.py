import time
import os
import mss
import requests
import pywinctl
import pyautogui
import mss.tools
import json
import subprocess
import argparse

from daemon.core.window import UniversalWindowObserver
from daemon.core.router import StateRouter
from shared.core.graph_db import GraphDatabase

class OmniOSDaemon:
    def __init__(self, debug_mode: bool = False):
        print(f"[Daemon] Booting Omni-OS v2 Core Systems... (Debug: {debug_mode})")
        self.debug_mode = debug_mode
        self.observer = UniversalWindowObserver()
        self.router = StateRouter()
        self.db = GraphDatabase()

    def _send_to_mapper(self, state_id: str, window_title: str, image_path: str):
        """Helper function to keep the network logic DRY."""
        try:
            with open(image_path, "rb") as img_file:
                response = requests.post(
                    "http://localhost:8002/map",
                    data={
                        "state_id": state_id, 
                        "window_title": window_title,
                        "debug": str(self.debug_mode).lower()  # Send the flag
                    },
                    files={"image": img_file}
                )
            if response.status_code == 200:
                print(f"[Daemon] Mapper Success: {response.json()['message']}")
            else:
                print(f"[Daemon] Mapper Failed: {response.text}")
        except requests.exceptions.ConnectionError:
            print("[Daemon] Error: Could not connect to Mapper on port 8001.")

    def _bootstrap_system(self):
        """Forces a clean mapping of the OS desktop and pre-warms target apps."""
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
        
    def run(self, user_goal: str):
        # Execute the startup sequence first
        self._bootstrap_system()

        print(f"\n[Daemon] Task Accepted: '{user_goal}'")
        print("[Daemon] Entering autonomous loop. Press Ctrl+C to abort.\n")
        
        step_count = 1
        
        try:
            while True:
                print(f"--- Step {step_count} ---")
                
                # Sense the Environment
                window_title = self.observer.get_active_window_title()
                
                # Identify the State
                state_id = self.router.resolve_state_id(window_title)
                print(f"[Daemon] Active Window: '{window_title}'")
                print(f"[Daemon] Resolved State: '{state_id}'")
                
                # Check Memory (Graph Database)
                state_graph = self.db.get_state(state_id)
                
                if state_graph:
                    # ==========================================
                    # THE FAST PATH (Gemini Flash Planner)
                    # ==========================================
                    print(f"[Daemon] -> CACHE HIT! State '{state_id}' is mapped.")
                    print(f"[Daemon] -> Triggering Planner microservice (Gemini Flash)...")
                    
                    # TODO: Send Image + state_graph.model_dump_json() to Planner
                    # TODO: Execute returned ActionPlan
                    
                    print("[Daemon] (Mocking Fast Path execution...)\n")
                    time.sleep(2) 
                    
                else:
                    # ==========================================
                    # THE SLOW PATH (Trigger Mapper Microservice)
                    # ==========================================
                    print(f"[Daemon] -> CACHE MISS! Unmapped state: '{state_id}'")
                    print(f"[Daemon] -> Taking screenshot and triggering Mapper microservice...")

                    # Get the exact bounding box of the focused window
                    active_window = pywinctl.getActiveWindow()
                    if active_window:
                        # Ensure we don't pass negative coordinates if the window is off-screen
                        monitor_bbox = {
                            "top": max(0, active_window.top),
                            "left": max(0, active_window.left),
                            "width": active_window.width,
                            "height": active_window.height
                        }
                    else:
                        # Fallback to full primary screen if no window is active
                        monitor_bbox = 1 

                    # Take a screenshot of ONLY that bounding box
                    screenshot_path = "tmp/temp_capture.png"
                    with mss.MSS() as sct:
                        # grab() takes the custom dictionary and returns an image object
                        img = sct.grab(monitor_bbox)
                        # Save it to disk
                        mss.tools.to_png(img.rgb, img.size, output=screenshot_path)

                    # Send the multipart/form-data request to the Mapper
                    self._send_to_mapper(state_id, window_title, screenshot_path)
                    
                    # Cleanup local screenshot
                    if os.path.exists(screenshot_path):
                        os.remove(screenshot_path)
                        
                    print("[Daemon] Slow path complete. Looping...\n")
                    time.sleep(2)
                
                step_count += 1
                
                # Mock termination after 3 steps for safety during testing
                if step_count > 3:
                    print("[Daemon] Mock limit reached. Exiting.")
                    break
                    
        except KeyboardInterrupt:
            print("\n[Daemon] Execution aborted by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Omni-OS Daemon Core")
    parser.add_argument("--debug", action="store_true", help="Enable visual debugging artifacts in shared/debug/")
    args = parser.parse_args()

    daemon = OmniOSDaemon(debug_mode=args.debug)
    daemon.run(user_goal="Open Firefox and search for Omni-OS.")