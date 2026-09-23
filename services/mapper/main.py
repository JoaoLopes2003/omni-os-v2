import os
import json
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import uvicorn
from dotenv import load_dotenv, find_dotenv
import shutil
import time

from services.mapper.mapper_agent import MapperAgent
from services.mapper.ocr import OCRProcessor
from services.mapper.debug_renderer import DebugRenderer

load_dotenv(find_dotenv())

app = FastAPI(title="Omni-OS Mapper Microservice")

# Initialize the agent (Expects GEMINI_API_KEY in the environment)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("[Warning] GEMINI_API_KEY environment variable is missing!")

mapper_agent = MapperAgent(api_key=api_key)
ocr_processor = OCRProcessor()
renderer = DebugRenderer()

@app.post("/map")
async def map_ui_state(
    image: UploadFile = File(...),
    state_id: str = Form(...),
    window_title: str = Form(...),
    debug: str = Form("false")  # Accept debug flag
):
    print(f"[Mapper API] Received mapping request for state: '{state_id}'")
    is_debug = debug.lower() == "true"
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
            content = await image.read()
            temp_img.write(content)
            temp_image_path = temp_img.name

        print("[Mapper API] Running EasyOCR...")
        ocr_data = ocr_processor.extract_normalized_data(temp_image_path)
        
        print(f"[Mapper API] Triggering Gemini 3.1 Pro...")
        final_state = mapper_agent.generate_state_map(
            image_path=temp_image_path,
            window_title=window_title,
            state_id=state_id,
            raw_ocr_data=ocr_data
        )

        # ==========================================
        # DEBUG ARTIFACT GENERATION
        # ==========================================
        if is_debug:
            timestamp = int(time.time())
            debug_dir = f"shared/debug/{state_id}_{timestamp}"
            os.makedirs(debug_dir, exist_ok=True)
            print(f"[Mapper API] Debug mode enabled. Saving artifacts to {debug_dir}/")
            
            # 1. Original Screenshot
            shutil.copy2(temp_image_path, f"{debug_dir}/1_original_screenshot.png")
            
            # 2. OCR JSON Result
            with open(f"{debug_dir}/2_ocr_data.json", "w") as f:
                json.dump(ocr_data, f, indent=2)
                
            # 3. OCR Bounding Boxes Image
            renderer.draw_ocr_bboxes(temp_image_path, ocr_data, f"{debug_dir}/3_ocr_visual.png")
            
            # 4. Gemini JSON Result
            with open(f"{debug_dir}/4_gemini_data.json", "w") as f:
                f.write(final_state.model_dump_json(indent=2))
                
            # 5. Gemini Center Dots Image
            renderer.draw_gemini_elements(temp_image_path, final_state.model_dump()["elements"], f"{debug_dir}/5_gemini_visual.png")

        os.remove(temp_image_path)

        return {
            "status": "success",
            "message": f"Successfully mapped {len(final_state.elements)} elements.",
            "data": final_state.model_dump()
        }

    except Exception as e:
        print(f"[Mapper API] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the server locally on port 8001
    uvicorn.run(app, host="0.0.0.0", port=8002)