from llm_sarsa_agent import LLMSARSA

def run_sanity_check():
    print("--- Starting AI Integration Sanity Check ---")
    
    # 1. Initialize the agent 
    # (Assuming 5 possible states and 3 possible VM actions for this tiny test)
    print("Initializing Agent and LLM Engine...")
    agent = LLMSARSA(num_states=5, num_actions=3, alpha=0.1, gamma=0.9, epsilon=0.1)
    
    # 2. Define a dummy cloud state
    # Let's simulate a good scenario: a tiny task going to a powerful, idle VM
    dummy_cloud_state = {
        "task_mi": 15000,
        "task_type": "tiny",
        "vm_cpu": 1.0,
        "vm_mem": 1.0,
        "vm_load": 10,
        "total_vms": 20
    }
    
    # 3. Define the RL step variables
    state_idx = 0
    action_idx = 1
    reward = 10  # A positive reward for what should be a good allocation
    next_state_idx = 1
    next_action_idx = 1
    
    print(f"Initial Q-Value at state {state_idx}, action {action_idx}: {agent.q_table[state_idx, action_idx]}")
    
    # 4. Run the update loop
    print("Triggering LLM-Guided SARSA Update (This may take a few seconds if the LLM is cold)...")
    new_q_value = agent.update(
        state_idx=state_idx, 
        action_idx=action_idx, 
        reward=reward, 
        next_state_idx=next_state_idx, 
        next_action_idx=next_action_idx, 
        cloud_state_dict=dummy_cloud_state
    )
    
    print(f"Updated Q-Value (Q**): {new_q_value}")
    
    # Verify the buffer worked
    state_key = agent.llm_engine._generate_state_key(
        dummy_cloud_state["task_mi"], dummy_cloud_state["vm_cpu"], 
        dummy_cloud_state["vm_mem"], dummy_cloud_state["vm_load"]
    )
    print(f"Cached Heuristic in Buffer: {agent.llm_engine.heuristic_buffer.get(state_key)}")
    print("--- Sanity Check Complete ---")

if __name__ == "__main__":
    run_sanity_check()
