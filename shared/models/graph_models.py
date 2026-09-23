from typing import List
from pydantic import BaseModel, Field

class MappedElement(BaseModel):
    """A unified node representing a physical action the agent can take strictly within THIS specific state."""
    id: str = Field(description="A descriptive string ID (e.g., 'btn_file', 'icon_settings', 'tab_terminal').")
    text: str = Field(description="The cleaned text of the element, or a brief description if it is an icon.")
    element_type: str = Field(default="button", description="Must be 'button', 'input', or 'icon'.")
    
    # --- Spatial Data ---
    center_x: int = Field(description="Normalized X (0-1000) for the physical center of the element.")
    center_y: int = Field(description="Normalized Y (0-1000) for the physical center of the element.")

class ScreenStateNode(BaseModel):
    """A self-contained graph node representing a unique, static UI state (The Chrome)."""
    state_id: str
    elements: List[MappedElement] = []