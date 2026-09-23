import os
import json
from typing import Optional
from shared.models.graph_models import ScreenStateNode

class GraphDatabase:
    """Manages persistence and retrieval of UI state graphs."""
    
    def __init__(self, db_dir: str = "shared/data/states"):
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)
        # In-memory cache for ultra-fast reads during the execution loop
        self._cache: dict[str, ScreenStateNode] = {}

    def _get_file_path(self, state_id: str) -> str:
        """Returns the absolute file path for a given state_id."""
        return os.path.join(self.db_dir, f"{state_id}.json")

    def get_state(self, state_id: str) -> Optional[ScreenStateNode]:
        """Retrieves a state from cache or disk. Returns None if unmapped."""
        # Check ultra-fast cache
        if state_id in self._cache:
            return self._cache[state_id]

        file_path = self._get_file_path(state_id)
        
        # Check disk storage
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    state_node = ScreenStateNode.model_validate(data)
                    self._cache[state_id] = state_node  # Populate cache
                    return state_node
            except Exception as e:
                print(f"[GraphDB] Error loading state '{state_id}': {e}")
                return None
                
        # Unmapped State
        return None

    def save_state(self, state_node: ScreenStateNode) -> bool:
        """Persists a new or updated state to disk and updates the cache."""
        self._cache[state_node.state_id] = state_node
        file_path = self._get_file_path(state_node.state_id)
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                # model_dump_json is native to Pydantic v2
                f.write(state_node.model_dump_json(indent=2))
            print(f"[GraphDB] Saved state '{state_node.state_id}' to disk.")
            return True
        except Exception as e:
            print(f"[GraphDB] Error saving state '{state_node.state_id}': {e}")
            return False

if __name__ == "__main__":
    # --- Dummy Test Block ---
    from shared.models.graph_models import MappedElement
    
    db = GraphDatabase()
    
    # Create a mock state matching our new schema
    mock_state = ScreenStateNode(
        state_id="vscode_main_editor",
        elements=[
            MappedElement(
                id="1", 
                text="File", 
                element_type="button", 
                center_x=30, 
                center_y=15, 
                leads_to_state="vscode_file_menu"
            ),
            MappedElement(
                id="cmd_palette", 
                text="Open Command Palette", 
                element_type="app_shortcut", 
                hotkey=["ctrl", "shift", "p"]
            )
        ]
    )
    
    # Save the state
    db.save_state(mock_state)
    
    # Retrieve the state
    retrieved = db.get_state("vscode_main_editor")
    if retrieved:
        print(f"Retrieved State: '{retrieved.state_id}'")
        for el in retrieved.elements:
            print(f" - {el.id}: {el.text} ({el.element_type})")