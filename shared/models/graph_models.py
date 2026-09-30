from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict

class BoundingBox(BaseModel):
    """Absolute coordinates for containers and dynamic grid items."""
    x: int
    y: int
    w: int
    h: int

class ElementNode(BaseModel):
    """An interactive or dynamic element inside a container."""
    id: str = Field(description="Unique snake_case identifier for the element")
    text: Optional[str] = Field(default=None, description="Visible text, if any")
    element_type: Literal['button', 'icon', 'input', 'text', 'image', 'dropdown'] = Field(...)
    description: str = Field(description="Semantic description of what this element does")
    
    # Normalized coordinates relative to the parent container (0.0 to 1.0)
    rel_x: float = Field(description="Center X relative to parent container width")
    rel_y: float = Field(description="Center Y relative to parent container height")
    
    is_dynamic: bool = Field(
        default=False, 
        description="True if the content changes frequently (e.g., playing song title, video player)"
    )
    
    # Predictive State Routing
    opens_container: Optional[str] = Field(
        default=None, 
        description="ID of the local container/overlay this reveals when clicked"
    )
    navigates_to_view: Optional[str] = Field(
        default=None, 
        description="ID of the new global ScreenView this navigates to when clicked"
    )

class TemplateElementNode(BaseModel):
    """An interactive part of a template card."""
    id: str = Field(description="Unique snake_case identifier")
    text: Optional[str] = Field(default=None, description="The text, ONLY if is_dynamic is False")
    element_type: Literal['button', 'icon', 'input', 'text', 'image', 'dropdown'] = Field(...)
    description: str = Field(description="Semantic role of the element")
    
    # Coordinates relative to the IDEALIZED single item, not the container
    rel_x: float = Field(description="Center X relative to idealized item width")
    rel_y: float = Field(description="Center Y relative to idealized item height")
    
    is_dynamic: bool = Field(default=True, description="False if this element is identical across all cards")

class TemplateNode(BaseModel):
    """A reusable structure for grid/list items."""
    template_id: str
    description: str
    collection_bounds: BoundingBox = Field(
        description="The outer bounding box of the entire region where these items are displayed."
    )
    elements: List[TemplateElementNode] = Field(
        description="Elements mapped with coordinates relative to a single idealized item's bounds"
    )

class ContainerNode(BaseModel):
    """A macro-section of the screen (e.g., Sidebar, Top Nav, Main Body)."""
    id: str = Field(description="Unique snake_case identifier for the container")
    description: str = Field(description="Semantic description of the container's purpose")
    
    bbox: BoundingBox = Field(description="Absolute OS coordinates of the container")
    
    scrollable: Literal['vertical', 'horizontal', 'both', 'none'] = Field(default='none')
    is_overlay: bool = Field(
        default=False, 
        description="True if this is a popup/dropdown that requires a click to be visible"
    )
    
    # Standard elements
    elements: List[ElementNode] = Field(default_factory=list)
    
    # Template properties
    item_template: Optional[TemplateNode] = Field(default=None)

class ScreenView(BaseModel):
    """The root Visual DOM for a specific application state."""
    view_id: str = Field(description="Unique identifier for this application state (e.g., spotify_home)")
    process_name: str = Field(description="The OS executable name (e.g., spotify.exe, chrome.exe)")
    description: str = Field(description="What is the user currently looking at and able to do here?")
    
    containers: List[ContainerNode] = Field(default_factory=list)