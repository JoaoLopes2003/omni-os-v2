import os
import tempfile
import uvicorn
from dotenv import load_dotenv, find_dotenv
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from services.planner.planner_agent import PlannerAgent

load_dotenv(find_dotenv())

app = FastAPI(title="Omni-OS Planner Microservice")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("[Warning] GEMINI_API_KEY environment variable is missing!")

planner_agent = PlannerAgent(api_key=api_key)

@app.post("/plan")
async def generate_action_plan(
    image: UploadFile = File(...),
    user_goal: str = Form(...),
    state_id: str = Form(...),
    state_graph: str = Form(...),
    open_windows: str = Form(...),
    action_history: str = Form("[]"),
    memory: str = Form("{}")  # Default is an empty JSON object
):
    print(f"[Planner API] Received planning request for state '{state_id}'")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
            content = await image.read()
            temp_img.write(content)
            temp_image_path = temp_img.name

        windows_list = json.loads(open_windows)
        history_list = json.loads(action_history)
        memory_dict = json.loads(memory)

        action_plan = planner_agent.generate_plan(
            image_path=temp_image_path,
            user_goal=user_goal,
            state_id=state_id,
            state_graph_json=state_graph,
            open_windows=windows_list,
            action_history=history_list,
            memory=memory_dict
        )

        os.remove(temp_image_path)

        return {
            "status": "success",
            "thought_process": action_plan.thought_process,
            "data": [action.model_dump() for action in action_plan.actions]
        }

    except Exception as e:
        print(f"[Planner API] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)