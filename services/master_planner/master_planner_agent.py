import json
from google import genai
from google.genai import types
from shared.models.action_models import TaskDecomposition

class MasterPlannerAgent:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-3.8-flash"

    def decompose_task(self, user_goal: str, user_profile: str) -> tuple[TaskDecomposition, str, dict[str, int]]:
        
        system_instruction = f"""You are the Omni-OS Master Planner.
Your job is to analyze complex user requests, extract explicit input variables, filter out irrelevant noise, and break the core objective into a sequence of semantic subgoals.

RULES:
1. NOISE FILTERING: Ignore conversational filler or facts that are irrelevant to achieving the core physical task.
2. ACTIONABLE VARIABLE EXTRACTION: Only extract explicit data provided in the prompt that needs to be typed into the UI (e.g., known email addresses, specific URLs).
3. EXPLICIT DATA PIPELINE (Supply & Demand): NEVER rely on the system clipboard to transfer data between subgoals, as the clipboard is volatile. If a subgoal involves copying data (like a link), explicitly instruct the Action Planner to save it to a variable (e.g., "Store the copied link in the variable `song_link`"). Later subgoals must reference that exact variable name. NEVER start a variable name with the prefix "temp_".
4. VERIFIABLE END CONDITIONS: A subgoal must have a clear end state. If the goal involves retrieving data, the end condition is successfully saving that data into a variable.
5. RETAIN SEMANTIC CONTEXT: Do not strip away context needed for execution. If sending a message to a father, the subgoal must mention the recipient is the father so the Action Planner knows what tone to use.
6. SUBGOAL GRANULARITY: Subgoals should be high-level milestones, NOT atomic UI clicks.
7. USER PREFERENCES: Strictly follow the USER PROFILE to determine which applications to use. If the profile states a preference (e.g., "Use Gmail in the web browser"), explicitly declare that application in your subgoal instead of using generic terms like "email client".

USER PROFILE:
{user_profile}

### EXAMPLES ###

User Prompt: "Go to my file containing my email addresses and get my fathers email. Then send him an email saying that I am ok. Also, it is raining today."
Output:
{{
  "thought_process": "The user wants to find an email address in a local file, then send an email to that address. I will mandate a variable to pass the email between subgoals. Based on the User Profile (assuming default web mail), I will specify the browser.",
  "initial_memory": [],
  "subgoals": [
    {{"description": "Open the file containing email addresses, locate the father's email address, and use extract_data to store it as 'fathers_email'."}},
    {{"description": "Open the web browser, navigate to Gmail, and write an informal, reassuring email to the father saying 'I am ok'. Use the variable 'fathers_email' as the recipient address, then send it."}}
  ]
}}

User Prompt: "Go to Spotify, chose a song that my boss might like (he was born in 1974) and send an email to him with the spotify link. He's email is sergio.h.s.lopes@gmail.com. He studied math in 5th grade and my cousin went to Asia, but they never met."
Output:
{{
  "thought_process": "The user wants to find a 1974 song on Spotify and email the link. The email is provided, so I extract it. I will explicitly tell the agent to save the copied link to a variable. The tone should be formal for a boss. I will enforce the user's preferred email client.",
  "initial_memory": [{{"key": "target_email", "value": "sergio.h.s.lopes@gmail.com"}}],
  "subgoals": [
    {{"description": "Open Spotify, search for a popular song from 1974 that a boss would like. Click the share/copy button, and use the save_clipboard_to_memory action to store the link in the variable 'spotify_link'."}},
    {{"description": "Open the web browser and navigate to Gmail. Compose a new email to the boss using the address stored in 'target_email'. Type a formal message, insert the link stored in the variable 'spotify_link', and send it."}}
  ]
}}
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

        full_prompt = f"{system_instruction}\n\nUSER GOAL: {user_goal}"

        # Extract token usage
        token_usage = {
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count
        }
        
        return decomposition, full_prompt, token_usage