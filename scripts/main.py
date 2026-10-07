from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn

# Import your actual agent
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

# --- Define the Pydantic Data Models ---
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
    global last_state_idx, last_action_idx
    
    print(f"\n[+] Received Task {request.task_id} ({request.task_mi} MI)")
    
    current_state_idx = 0 
    
    # 1. Agent chooses the next VM (action) based on the Q-table
    chosen_action_idx = int(agent.choose_action(current_state_idx))
    
    # 2. If this isn't the first task, update the Q-table using the previous transition
    if last_state_idx is not None and last_action_idx is not None:
        target_vm = request.vms[chosen_action_idx] if request.vms else None
        
        cloud_state_dict = {
            "task_mi": request.task_mi,
            "task_type": request.task_type,
            "vm_cpu": target_vm.cpu_capacity if target_vm else 0.0,
            "vm_mem": target_vm.ram_capacity if target_vm else 0.0,
            "vm_load": target_vm.current_load_percent if target_vm else 0.0,
            "total_vms": len(request.vms)
        }
        
        # Fire the Q-table update using the LLM heuristic
        agent.update(
            state_idx=last_state_idx,
            action_idx=last_action_idx,
            reward=request.last_reward,
            next_state_idx=current_state_idx,
            next_action_idx=chosen_action_idx,
            cloud_state_dict=cloud_state_dict
        )
        print(f"[+] Q-Table Updated for State {last_state_idx}, Action {last_action_idx}")
    
    # 3. Save current state/action for the next API call
    last_state_idx = current_state_idx
    last_action_idx = chosen_action_idx
    
    print(f"[+] AI Decision: Assigning to VM {chosen_action_idx}")
    
    return {
        "assigned_vm_id": chosen_action_idx,
        "status": "success"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)