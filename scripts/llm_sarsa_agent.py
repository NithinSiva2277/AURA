import numpy as np
from heuristic_generator import LLMHeuristicGenerator

class LLMSARSA:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha       # Learning Rate
        self.gamma = gamma       # Discount Factor
        self.epsilon = epsilon   # Exploration rate
        
        # Initialize the Q-table to zeros
        self.q_table = np.zeros((num_states, num_actions))
        
        # Initialize our LLM Heuristic Engine
        self.llm_engine = LLMHeuristicGenerator(model_name="llama3.2:3b")

    def choose_action(self, state_idx):
        """ Epsilon-greedy policy for action selection """
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.q_table[state_idx, :])

    def _calculate_l2_loss(self, q_star, q_current, d_llm):
        """ The L2 loss function from the paper to mitigate hallucination. """
        error = q_star - q_current
        raw_penalty = error * d_llm * ((q_star - q_current) ** 2)
        # GUARDRAIL: Clip penalty to prevent exploding Q-values
        return np.clip(raw_penalty, -10.0, 10.0)

    def update(self, state_idx, action_idx, reward, next_state_idx, next_action_idx, cloud_state_dict):
        """ The core learning loop that updates the Q-table using the LLM's guidance. """
        q_current = self.q_table[state_idx, action_idx]
        q_next = self.q_table[next_state_idx, next_action_idx]
        
        d_llm = self.llm_engine.get_heuristic(
            task_mi=cloud_state_dict["task_mi"],
            task_type=cloud_state_dict["task_type"],
            vm_cpu=cloud_state_dict["vm_cpu"],
            vm_mem=cloud_state_dict["vm_mem"],
            vm_load=cloud_state_dict["vm_load"],
            total_vms=cloud_state_dict["total_vms"]
        )
        
        td_error = reward + (self.gamma * q_next) - q_current
        q_star = q_current + (self.alpha * td_error) + d_llm
        
        l2_loss = self._calculate_l2_loss(q_star, q_current, d_llm)
        q_double_star = l2_loss + q_star
        q_double_star = np.clip(q_double_star, -100.0, 100.0)
        
        self.q_table[state_idx, action_idx] = q_double_star
        return q_double_star