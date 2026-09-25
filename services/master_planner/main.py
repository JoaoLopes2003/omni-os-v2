import os
import tempfile
import uvicorn
from dotenv import load_dotenv, find_dotenv
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from services.master_planner.master_planner_agent import MasterPlannerAgent

load_dotenv(find_dotenv())

app = FastAPI(title="Omni-OS Master Planner Microservice")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("[Warning] GEMINI_API_KEY environment variable is missing!")

master_planner_agent = MasterPlannerAgent(api_key=api_key)

@app.post("/decompose")
async def decompose_user_goal(
    user_goal: str = Form(...),
    user_profile: str = Form("No specific preferences provided.") # Default fallback
):
    print(f"[Planner API] Received Master Plan request for goal.")
    
    try:
        decomposition, raw_prompt, token_usage = master_planner_agent.decompose_task(user_goal, user_profile)

        return {
            "status": "success",
            "thought_process": decomposition.thought_process,
            "initial_memory": decomposition.initial_memory,
            "subgoals": [sub.model_dump() for sub in decomposition.subgoals],
            "raw_prompt": raw_prompt,
            "token_usage": token_usage
        }

    except Exception as e:
        print(f"[Planner API] Master Planner Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)