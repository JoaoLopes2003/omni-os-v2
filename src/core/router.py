import json
import re
import os

class StateRouter:
    def __init__(self, config_path: str = "config/state_routing.json"):
        self.routes = []
        self._load_config(config_path)

    def _load_config(self, config_path: str):
        if not os.path.exists(config_path):
            print(f"[Router] Warning: Config not found at {config_path}. Starting with empty routes.")
            return
            
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Compile regex patterns on startup for zero-latency routing later
            for route in data.get("routes", []):
                self.routes.append({
                    "pattern": re.compile(route["pattern"], re.IGNORECASE),
                    "state_id": route["state_id"]
                })

    def get_state_id(self, window_title: str) -> str:
        """
        Evaluates the window title against known patterns.
        Returns the mapped state_id, or 'unmapped_state' if no match is found.
        """
        for route in self.routes:
            if route["pattern"].match(window_title) or route["pattern"].search(window_title):
                return route["state_id"]
                
        return "unmapped_state"

if __name__ == "__main__":
    # Quick Test
    router = StateRouter()
    
    test_titles = [
        "window.py - Post University - Visual Studio Code",
        "Spotify Premium",
        "Desktop",
        "Omni-OS Multimodal GUI Agent Architecture - Google Gemini — Mozilla Firefox",
        "Unknown Application 2026"
    ]
    
    for title in test_titles:
        state = router.get_state_id(title)
        print(f"Title: '{title}' ---> State ID: '{state}'")