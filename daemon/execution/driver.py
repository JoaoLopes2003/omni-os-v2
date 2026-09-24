import time
import pyautogui
import pyperclip
from shared.models.schemas import UIElement, BoundingBox
from services.planner.llm_planner import ActionPlan

class ActionDriver:
    def __init__(self, primary_button: str = "left"):
        """
        Initializes the driver.
        :param primary_button: Which mouse button acts as the primary click. 
                               Defaulting to "right" here due to your OS-level inversion.
        """
        # Mandatory failsafe: moving mouse to any screen corner kills execution
        pyautogui.FAILSAFE = True
        # Add a tiny pause between actions to mimic human interaction
        pyautogui.PAUSE = 0.2

        self.primary_button = primary_button

    def execute(
            self,
            plan: ActionPlan,
            elements: dict[int, UIElement],
            screen_width: int,
            screen_height: int,
            offset_x: int = 0,
            offset_y: int = 0
        ) -> bool:
        """
        Executes the planned action on the physical OS.
        Returns True if successful, False if the action cannot be performed.
        """
        # Handle 'done' action
        if plan.action == "done":
            print(f"[Driver] Task marked as completed. Reasoning: {plan.reasoning}")
            return True

        # Handle 'wait' action
        if plan.action == "wait":
            print(f"[Driver] System loading/transitioning. Agent is waiting... Reasoning: {plan.reasoning}")
            time.sleep(4.0) 
            return True
            
        # Handle 'type' action (Atomic Keyboard Input via Clipboard)
        if plan.action == "type":
            print(f"[Driver] Injecting text via clipboard: '{plan.text_input}'...")
            
            # Copy the exact string to the system clipboard
            pyperclip.copy(plan.text_input)
            
            # Add a microscopic pause to let the OS clipboard buffer catch up
            time.sleep(0.1)
            
            # Simulate a standard paste command (Ctrl + V)
            pyautogui.hotkey("ctrl", "v")
            
            # Clear the clipboard for security (optional but good practice)
            time.sleep(0.1)
            pyperclip.copy("")
            
            return True

        # Handle 'click_point' action
        if plan.action == "click_point":
            if plan.point_x < 0 or plan.point_y < 0:
                print(f"[Driver] Error: Invalid point coordinates received: ({plan.point_x}, {plan.point_y})")
                return False
                
            local_x = int((plan.point_x / 1000.0) * screen_width)
            local_y = int((plan.point_y / 1000.0) * screen_height)
            
            global_x = local_x + offset_x
            global_y = local_y + offset_y
            
            print(f"[Driver] Pointing at un-tagged element. Clicking Global({global_x}, {global_y})")
            pyautogui.click(global_x, global_y, duration=0.5, button=self.primary_button)
            return True

        # --- TAGGED ELEMENTS ONLY (click) ---
        target_element = elements.get(plan.element_id)
        if not target_element:
            print(f"[Driver] Error: Element ID {plan.element_id} not found.")
            return False

        local_center_x, local_center_y = target_element.bbox.get_absolute_center(screen_width, screen_height)
        global_center_x = local_center_x + offset_x
        global_center_y = local_center_y + offset_y

        if plan.action == "click":
            print(f"[Driver] Clicking element {plan.element_id} at ({global_center_x}, {global_center_y})")
            pyautogui.click(global_center_x, global_center_y, duration=0.5, button=self.primary_button)
            return True

        print(f"[Driver] Unsupported action: {plan.action}")
        return False

if __name__ == "__main__":
    # --- Dummy Test Block ---
    print("Testing OS Action Driver... Keep your hands off the mouse!")
    
    # Instantiate the driver
    driver = ActionDriver()
    
    # Create dummy screen dimensions (assume 1080p)
    screen_w, screen_h = 1920, 1080

    # Because pyautogui operates on the OS's Virtual Desktop coordinate system
    offset_x = 0  
    offset_y = 0
    
    # Create a dummy elements dictionary
    # We will place a fake button at 10% x and 10% y
    dummy_elements = {
        8: UIElement(
            element_id=8,
            bbox=BoundingBox(xmin=0.1, ymin=0.1, xmax=0.15, ymax=0.15),
            text="Firefox",
            confidence=0.99
        )
    }
    
    # Create a dummy click plan
    dummy_plan = ActionPlan(
        reasoning="Testing the driver execution.",
        action="click",
        element_id=8,
        text_input=""
    )
    
    # Execute! (You should see your mouse smoothly move to the top left quadrant and click)
    time.sleep(2) # Give you 2 seconds to move your hands away
    success = driver.execute(
        dummy_plan,
        dummy_elements,
        screen_w,
        screen_h, 
        offset_x, 
        offset_y
    )
    
    print(f"Driver execution successful: {success}")