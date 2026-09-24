from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class Action(BaseModel):
    """A single atomic operation to be executed by the OS Daemon."""
    action_type: Literal[
        'click_element', 
        'click_coordinate', 
        'type_text', 
        'hotkey', 
        'run_cli_command', 
        'extract_data', 
        'switch_window', 
        'wait', 
        'done', 
        'abort'
    ] = Field(description="The specific type of action to perform.")
    
    target_id: Optional[str] = Field(
        default=None, 
        description="The ID of the static element to click (e.g., 'btn_file'). Use only with 'click_element'."
    )
    
    target_x: Optional[int] = Field(
        default=None, 
        description="Normalized X coordinate (0-1000). Use only with 'click_coordinate'."
    )
    target_y: Optional[int] = Field(
        default=None, 
        description="Normalized Y coordinate (0-1000). Use only with 'click_coordinate'."
    )
    
    text_payload: Optional[str] = Field(
        default=None, 
        description="String to type, CLI command, extracted data, or exact window title. Use with 'type_text', 'run_cli_command', 'extract_data', or 'switch_window'."
    )
    memory_label: Optional[str] = Field(
        default=None,
        description="A descriptive name for the variable being saved (e.g., 'Invoice Total'). Use only with 'extract_data'."
    )
    
    hotkey_combo: Optional[List[str]] = Field(
        default=None, 
        description="List of keys to press together (e.g., ['ctrl', 's']). Use only with 'hotkey'."
    )
    
    wait_seconds: Optional[int] = Field(
        default=None,
        description="Seconds to wait for UI to load. Use only with 'wait'."
    )

class ActionPlan(BaseModel):
    """The complete response from the Planner."""
    thought_process: str = Field(
        description="A brief explanation of what the user wants and the logic behind the chosen actions."
    )
    actions: List[Action] = Field(
        description="The sequence of actions to execute in order."
    )

class Subgoal(BaseModel):
    description: str = Field(
        description="A clear, semantic milestone for the Action Planner to achieve (e.g., 'Switch to Spotify and copy the link to a 1974 song')."
    )

class MemoryItem(BaseModel):
    key: str = Field(description="The variable name (e.g., 'target_email')")
    value: str = Field(description="The extracted value (e.g., 'sergio.h.s.lopes@gmail.com')")

class TaskDecomposition(BaseModel):
    """The complete response from the Master Planner."""
    thought_process: str = Field(
        description="Analysis of the user prompt: identifying noise to ignore, variables to extract, and the logical breakdown."
    )
    initial_memory: List[MemoryItem] = Field(
        description="A list of explicit data extracted from the prompt."
    )
    subgoals: List[Subgoal] = Field(
        description="The chronological sequence of subgoals required to achieve the overall objective."
    )