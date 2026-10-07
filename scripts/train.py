import numpy as np
from cloud_env import CloudEnvironment
from llm_sarsa_agent import LLMSARSA

def train_agent():
    print("--- Starting LLM-Guided SARSA Training Phase ---")
    
    # 1. Initialize the World and the Agent
    num_vms = 20
    env = CloudEnvironment(num_vms=num_vms)
    
    # We discretize the cloud load into 10 possible states (0-10%, 10-20%, etc.)
    # The agent has 20 possible actions (choosing VM 0 through VM 19)
    agent = LLMSARSA(num_states=10, num_actions=num_vms, alpha=0.1, gamma=0.9, epsilon=0.2)
    
    # 2. Define the Training Loop (Episodes)
    num_episodes = 5  # Keeping it small for a quick test
    tasks_per_episode = 10 
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        state_idx = env.reset()
        
        # We need an initial action to start SARSA
        action_idx = agent.choose_action(state_idx)
        
        episode_reward = 0
        
        for step in range(tasks_per_episode):
            # Simulate a random incoming task from the Google dataset ranges
            # Example: 15,000 MI to 150,000 MI
            incoming_task_mi = np.random.randint(15000, 150000)
            
            # The agent takes the action and the environment reacts
            next_state_idx, reward, cloud_state_dict = env.step(action_idx, incoming_task_mi)
            episode_reward += reward
            
            # The agent chooses the NEXT action (required for SARSA math)
            next_action_idx = agent.choose_action(next_state_idx)
            
            # Update the Q-table using the LLM heuristic and L2 loss
            agent.update(
                state_idx=state_idx, 
                action_idx=action_idx, 
                reward=reward, 
                next_state_idx=next_state_idx, 
                next_action_idx=next_action_idx,
                cloud_state_dict=cloud_state_dict
            )
            
            # Move to the next step
            state_idx = next_state_idx
            action_idx = next_action_idx
            
            print(f"Step {step+1} | Task: {incoming_task_mi} MI -> Assigned to VM {action_idx} | Reward: {reward:.2f}")

        print(f"Total Reward for Episode {episode + 1}: {episode_reward:.2f}")

    print("\n--- Training Complete ---")
    
if __name__ == "__main__":
    train_agent()
