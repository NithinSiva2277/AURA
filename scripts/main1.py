from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn

# We will import our actual agent here later
from llm_sarsa_agent import LLMSARSA

app = FastAPI(title="LLM-SARSA Cloud Orchestrator API")

# --- Initialize the Brain ---
NUM_STATES = 1 
NUM_ACTIONS = 4
# This will trigger the LLM initialization on startup
agent = LLMSARSA(num_states=NUM_STATES, num_actions=NUM_ACTIONS)

# Track transitions for the SARSA update
last_state_idx = None
last_action_idx = None

# --- Define the Pydantic Data Models (Matches our JSON Schema) ---
class VMState(BaseModel):
    vm_id: int
    cpu_capacity: float
    ram_capacity: float
    current_load_percent: float

class CloudStateRequest(BaseModel):
    task_id: int
    task_mi: float
    task_type: str
    last_reward: float
    vms: List[VMState]

# --- The Core API Endpoint ---
@app.post("/schedule_task")
async def schedule_task(request: CloudStateRequest):
    print(f"\n[+] Received Task {request.task_id} ({request.task_mi} MI)")
    print(f"[+] Last Action Reward: {request.last_reward}")
    
    # 1. TODO: Pass request.last_reward to agent to update Q-table
    # agent.update(reward=request.last_reward, ...)
    
    # 2. TODO: Pass request.vms to LLM Heuristic Generator to get D_LLM
    # d_llm = llm_engine.get_heuristic(...)
    
    # 3. TODO: Agent chooses best VM based on reshaped Q* value
    # chosen_vm_id = agent.choose_action(...)
    
    # --- UPDATED DUMMY LOGIC WITH SAFETY CHECK ---
    if not request.vms:
        print("[!] Warning: Empty VM list received from CloudSim. Defaulting to VM 0.")
        chosen_vm_id = 0
    else:
        chosen_vm_id = request.vms[0].vm_id
    
    print(f"[+] AI Decision: Assigning to VM {chosen_vm_id}")
    
    return {
        "assigned_vm_id": chosen_vm_id,
        "status": "success"
    }
if __name__ == "__main__":
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
