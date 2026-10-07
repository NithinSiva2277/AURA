import numpy as np

class CloudEnvironment:
    def __init__(self, num_vms=20):
        self.num_vms = num_vms
        
        # Initialize VMs (CPU, RAM, Current Load, Cost per hour)
        # For simplicity in this step, we'll give them random baseline capacities
        self.vms = {
            i: {"cpu": np.random.uniform(0.5, 1.0), 
                "mem": np.random.uniform(0.5, 1.0), 
                "load": 0.0,
                "cost_rate": np.random.uniform(0.2, 1.0)} 
            for i in range(self.num_vms)
        }
        
        self.current_task = None
        self.total_makespan = 0
        self.total_cost = 0
        
    def reset(self):
        """Resets the cloud cluster back to zero load"""
        for i in range(self.num_vms):
            self.vms[i]["load"] = 0.0
        self.total_makespan = 0
        self.total_cost = 0
        return self._get_state()
        
    def _get_state(self):
        """Returns the current state of the cloud to feed into the RL Agent"""
        # A simple state representation: average load of the cluster
        avg_load = sum(vm["load"] for vm in self.vms.values()) / self.num_vms
        # Discretize the state into 10 buckets (0-10%, 10-20%, etc.) for the Q-table
        state_idx = min(int(avg_load / 10), 9)
        return state_idx

    def step(self, action_vm_id, task_mi):
        """
        The agent chooses a VM (action). We calculate the physical result.
        """
        target_vm = self.vms[action_vm_id]
        
        # Calculate how much load this task adds based on VM's CPU capacity
        # (A 15,000 MI task hits a weak CPU harder than a strong CPU)
        load_increase = (task_mi / 100000) / target_vm["cpu"]
        target_vm["load"] += load_increase
        
        # --- Calculate Reward (The core optimization math) ---
        reward = 0
        
        # 1. Penalize if the VM overloads (Load > 100%)
        if target_vm["load"] > 100.0:
            reward -= 50  # Massive penalty for crashing a server
            target_vm["load"] = 100.0 # Cap it
        else:
            reward += 10  # Good allocation
            
        # 2. Penalize High Cost
        reward -= (target_vm["cost_rate"] * 5)
        
        # 3. Calculate Degree of Imbalance (DI) Penalty
        loads = [vm["load"] for vm in self.vms.values()]
        imbalance = max(loads) - min(loads)
        reward -= (imbalance * 0.1) # Keep the cluster balanced!
        
        # Get the new state of the cluster
        next_state_idx = self._get_state()
        
        # Return physical stats for the LLM prompt
        cloud_state_dict = {
            "task_mi": task_mi,
            "task_type": "medium" if task_mi > 50000 else "tiny",
            "vm_cpu": target_vm["cpu"],
            "vm_mem": target_vm["mem"],
            "vm_load": target_vm["load"],
            "total_vms": self.num_vms
        }
        
        return next_state_idx, reward, cloud_state_dict
