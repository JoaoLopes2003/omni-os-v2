import time
import cv2
import os
import json
from src.core.capture import ScreenCaptureEngine
from src.core.schemas import BoundingBox, UIElement
from src.perception.ocr import TextPerceptionEngine
from src.perception.som import SomRenderer
from src.reasoning.llm_planner import OSPlannerEngine
from src.execution.driver import ActionDriver

def main():
    print("Initializing Omni-OS Engines...")
    capture_engine = ScreenCaptureEngine()
    ocr_engine = TextPerceptionEngine()
    renderer = SomRenderer()
    planner = OSPlannerEngine()
    driver = ActionDriver()

    # Create a debug directory if it doesn't exist
    os.makedirs("./src/debug", exist_ok=True)

    user_prompt = input("Enter your command for the OS Agent: ")
    action_history = []
    step_count = 1
    while True:
        print(f"\n--- [Step {step_count}] ---")
        
        # --- 1. PERCEPTION ---
        t0 = time.time()
        print("Capturing live screen...")
        frame = capture_engine.grab_frame()
        screen_height, screen_width, _ = frame.shape
        
        # [DEBUG] Save the raw, unedited frame
        cv2.imwrite(f"./src/debug/step{step_count}_01_raw_capture.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        offset_x = capture_engine.monitor["left"]
        offset_y = capture_engine.monitor["top"]
        
        print("Extracting text...")
        raw_results = ocr_engine.extract_text(frame)
        ui_elements = {}
        
        # [DEBUG] Log the raw OCR extraction to text
        with open(f"./src/debug/step{step_count}_02_ocr_elements.txt", "w") as f:
            for element_id, result in enumerate(raw_results, start=1):
                raw_bbox, text, confidence = result["bbox"], result["text"], result["confidence"]
                
                bbox = BoundingBox(
                    xmin = raw_bbox[0] / screen_width, 
                    ymin = raw_bbox[1] / screen_height, 
                    xmax = raw_bbox[2] / screen_width, 
                    ymax = raw_bbox[3] / screen_height
                )
                
                ui_elements[element_id] = UIElement(
                    element_id=element_id, bbox=bbox, text=text, confidence=confidence
                )
                f.write(f"ID: {element_id} | Text: '{text}' | BBox: ({bbox.xmin:.2f}, {bbox.ymin:.2f}, {bbox.xmax:.2f}, {bbox.ymax:.2f})\n")

        print("Rendering Set-of-Marks...")
        annotated_frame = renderer.draw_marks(frame, list(ui_elements.values()))
        perception_time = time.time() - t0
        
        # [DEBUG] Save the annotated SOM frame
        output_path = f"./src/debug/step{step_count}_03_som_annotated.png"
        cv2.imwrite(output_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        
        # --- 2. REASONING ---
        t1 = time.time()
        print("Asking LLM Planner for next action...")
        plan, raw_prompt = planner.plan_action(output_path, ui_elements, user_prompt, action_history)
        reasoning_time = time.time() - t1

        # [DEBUG] Log the exact prompt sent to the LLM
        with open(f"./src/debug/step{step_count}_04a_raw_prompt.txt", "w") as f:
            f.write(raw_prompt)
        
        # [DEBUG] Log the LLM's exact response
        with open(f"./src/debug/step{step_count}_04_llm_plan.txt", "w") as f:
            f.write(f"Reasoning: {plan.reasoning}\n")
            f.write(f"Action: {plan.action}\n")
            f.write(f"Target ID: {plan.element_id}\n")
            f.write(f"Text Input: {plan.text_input}\n")
            f.write(f"Point: ({plan.point_x}, {plan.point_y})\n")
            
        print(f"Action: {plan.action} | Target ID: {plan.element_id}")
        
        # --- 3. EXECUTION ---
        t2 = time.time()
        if plan.action == "done":
            print("The user request has been fulfilled!")
            break

        if plan.action == "abort":
            print(f"ABORTED: The agent stopped the task. Reasoning: {plan.reasoning}")
            break
        
        success = driver.execute(plan, ui_elements, screen_width, screen_height, offset_x, offset_y)
        execution_time = time.time() - t2

        # [DEBUG] Save timing metrics
        timings = {
            "perception_sec": round(perception_time, 3),
            "reasoning_sec": round(reasoning_time, 3),
            "execution_sec": round(execution_time, 3),
            "total_step_sec": round(perception_time + reasoning_time + execution_time, 3)
        }
        with open(f"./src/debug/step{step_count}_05_timings.json", "w") as f:
            json.dump(timings, f, indent=4)
        
        # --- 4. STATE TRANSITION ---
        time.sleep(2)
        action_history.append(
            f"Step {step_count}: Action '{plan.action}' on Target ID {plan.element_id}. Reasoning: '{plan.reasoning}'"
        )
        step_count += 1

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()