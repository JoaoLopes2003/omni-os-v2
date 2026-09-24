import json
from google import genai
from google.genai import types
from shared.models.action_models import TaskDecomposition

class MasterPlannerAgent:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-3.8-flash"

    def decompose_task(self, user_goal: str) -> TaskDecomposition:
        
        system_instruction = """You are the Omni-OS Master Planner.
Your job is to analyze complex user requests, extract explicit input variables, filter out irrelevant noise, and break the core objective into a sequence of semantic subgoals.

RULES:
1. NOISE FILTERING: Ignore conversational filler or facts that are irrelevant to achieving the core physical task.
2. ACTIONABLE VARIABLE EXTRACTION: Only extract explicit data provided in the prompt that needs to be typed into the UI (e.g., known email addresses, specific URLs).
3. EXPLICIT DATA PIPELINE (Supply & Demand): If a subgoal requires finding information to be used later, explicitly name the variable to store it in (e.g., "Store the result as `extracted_email`"). Later subgoals must reference that exact variable name. NEVER start a variable name with the prefix "temp_".
4. RETAIN SEMANTIC CONTEXT: Do not strip away context needed for execution. If sending a message to a father, the subgoal must mention the recipient is the father so the Action Planner knows what tone to use.
5. SUBGOAL GRANULARITY: Subgoals should be high-level milestones, NOT atomic UI clicks.
    - BAD: "Click the windows key, type spotify, press enter, click search bar..." (Too atomic)
    - GOOD: "Open Spotify, search for a popular song from 1974, and copy its share link to the clipboard." (Perfect)

### EXAMPLES ###

User Prompt: "Go to my file containing my email addresses and get my fathers email. Then send him an email saying that I am ok. Also, it is raining today."
Output:
{
  "thought_process": "The user wants to find an email address in a local file, then send an email to that address. The weather 'raining today' is irrelevant noise. I need to mandate a variable to pass the email between subgoals. The tone of the email should reflect a child writing to their father.",
  "initial_memory": [],
  "subgoals": [
    {"description": "Open the file containing email addresses, locate the father's email address, and use extract_data to store it as 'fathers_email'."},
    {"description": "Open the email client and write an informal, reassuring email to the father saying 'I am ok'. Use the variable 'fathers_email' as the recipient address, then send it."}
  ]
}

User Prompt: "Go to Spotify, chose a song that my boss might like (he was born in 1974) and send an email to him with the spotify link. He's email is sergio.h.s.lopes@gmail.com. He studied math in 5th grade and my cousin went to Asia, but they never met."
Output:
{
  "thought_process": "The user wants to find a 1974 song on Spotify and email the link. The email is explicitly provided, so I will extract it. The math and cousin details are irrelevant noise. The tone should be suitable for a boss.",
  "initial_memory": [{"key": "target_email", "value": "sergio.h.s.lopes@gmail.com"}],
  "subgoals": [
    {"description": "Open Spotify, search for a popular song for someone born in 1974 that a boss would like, and copy its share link to the clipboard."},
    {"description": "Open the email client, compose a new email to the boss using the address stored in 'target_email'. Paste the Spotify link from the clipboard into the body, add a formal message, and send it."}
  ]
}
#################

Break the following USER GOAL into a logical TaskDecomposition."""

        print(f"[MasterPlanner] Decomposing complex goal...")

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[system_instruction, f"USER GOAL: {user_goal}"],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TaskDecomposition,
                temperature=0.1 
            )
        )
        
        decomposition = TaskDecomposition.model_validate_json(response.text)
        print(f"[MasterPlanner] Extracted {len(decomposition.initial_memory)} variables and generated {len(decomposition.subgoals)} subgoals.")
        return decomposition