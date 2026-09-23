import os
import json
import re

class StateRouter:
    """Routes raw OS window titles to canonical State IDs using Regex."""
    
    def __init__(self, config_path: str = "shared/config/state_routing.json"):
        self.routes = []
        self._load_config(config_path)

    def _load_config(self, config_path: str):
        if not os.path.exists(config_path):
            print(f"[Router] Warning: Config not found at {config_path}. Creating default.")
            self._create_default_config(config_path)
            
        with open(config_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                # Compile regex patterns on startup for maximum speed during execution loop
                for route in data.get("routes", []):
                    self.routes.append({
                        "pattern": re.compile(route["pattern"], re.IGNORECASE),
                        "state_id": route["state_id"]
                    })
            except json.JSONDecodeError:
                print(f"[Router] Error: Invalid JSON in {config_path}")

    def _create_default_config(self, config_path: str):
        """Creates a dummy config if none exists."""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        default_data = {
            "routes": [
                {"pattern": "Desktop", "state_id": "os_desktop"}
            ]
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_data, f, indent=2)
        self.routes.append({
            "pattern": re.compile("Desktop", re.IGNORECASE),
            "state_id": "os_desktop"
        })

    def resolve_state_id(self, window_title: str) -> str:
        """
        Evaluates the window title against known patterns.
        Returns the mapped state_id, or 'unmapped_app' if no match is found.
        """
        for route in self.routes:
            if route["pattern"].match(window_title) or route["pattern"].search(window_title):
                return route["state_id"]
                
        return "unmapped_app"

if __name__ == "__main__":
    # --- Dummy Test Block ---
    router = StateRouter()
    
    test_titles = [
        "window.py - Post University - Visual Studio Code",
        "Spotify Premium",
        "Desktop",
        "Omni-OS Multimodal GUI Agent Architecture - Google Gemini — Mozilla Firefox",
        "Calculator"
    ]
    
    print("Testing State Router:")
    for title in test_titles:
        state = router.resolve_state_id(title)
        print(f" Raw Title: '{title}'\n -> Routed State: '{state}'\n")