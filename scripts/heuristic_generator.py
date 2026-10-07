import json
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_core.exceptions import OutputParserException

class LLMHeuristicGenerator:
    def __init__(self, model_name="llama3.2:3b"):
        """
        Initializes the LLM and the Few-Shot Prompt Template.
        We are defaulting to a local Llama-3 model via Ollama to save API costs during training.
        """
        print(f"Initializing LLM Heuristic Engine with model: {model_name}...")
        
        # Connect to local LLM (Make sure Ollama is running on your Ubuntu machine)
        self.llm = OllamaLLM(model=model_name, temperature=0.1) # Low temp for deterministic, logical outputs
        
        # Define the exact prompt structure requested by the research paper
        self.prompt_template = PromptTemplate(
            input_variables=["task_mi", "task_type", "vm_cpu", "vm_mem", "vm_load", "total_vms"],
            template="""
            [SYSTEM PROMPT]
            You are a Cloud Resource Optimization Heuristic Evaluator. 
            Evaluate how well the proposed Task fits the target Virtual Machine (VM).
            Output ONLY a valid JSON object with a single float field "heuristic_value" between 0.0 (terrible match / high overload risk) and 1.0 (optimal resource utilization and low latency). Do not include any markdown formatting, backticks, or extra text.

            [FEW-SHOT EXAMPLES]
            State: Task size = 15000 MI, Task Type = tiny, Target VM = CPU 0.5, RAM 0.03, Current Load = 10%, Total VMs = 20
            Evaluation: {{"heuristic_value": 0.95}}

            State: Task size = 300000 MI, Task Type = huge, Target VM = CPU 0.5, RAM 0.03, Current Load = 85%, Total VMs = 20
            Evaluation: {{"heuristic_value": 0.05}}

            State: Task size = 120000 MI, Task Type = large, Target VM = CPU 1.0, RAM 1.0, Current Load = 40%, Total VMs = 20
            Evaluation: {{"heuristic_value": 0.80}}

            [INPUT QUERY]
            State: Task size = {task_mi} MI, Task Type = {task_type}, Target VM = CPU {vm_cpu}, RAM {vm_mem}, Current Load = {vm_load}%, Total VMs = {total_vms}
            Evaluation:
            """
        )

        # Create a basic memory buffer (D(G(p)) from the paper) to cache state evaluations
        self.heuristic_buffer = {}

    def _generate_state_key(self, task_mi, vm_cpu, vm_mem, vm_load):
        """ Creates a unique string key for the caching buffer """
        # Rounding load to nearest 5% to group similar states and increase cache hits
        rounded_load = round(vm_load / 5.0) * 5.0
        return f"{task_mi}_{vm_cpu}_{vm_mem}_{rounded_load}"

    def get_heuristic(self, task_mi, task_type, vm_cpu, vm_mem, vm_load, total_vms=20):
        """
        Retrieves the heuristic value. Checks the buffer first, calls the LLM if not found.
        """
        state_key = self._generate_state_key(task_mi, vm_cpu, vm_mem, vm_load)
        
        # 1. Check the Heuristic Buffer D(G(p)) first! (Saves massive compute time)
        if state_key in self.heuristic_buffer:
            return self.heuristic_buffer[state_key]
        
        # 2. Format the prompt with current cloud state
        prompt = self.prompt_template.format(
            task_mi=task_mi,
            task_type=task_type,
            vm_cpu=vm_cpu,
            vm_mem=vm_mem,
            vm_load=vm_load,
            total_vms=total_vms
        )
        
        # 3. Call the LLM
        try:
            response = self.llm.invoke(prompt)
            
            # Clean up the response in case the LLM tries to be chatty
            clean_response = response.strip().replace("```json", "").replace("```", "")
            
            # 4. Parse the JSON to extract D_LLM
            result_dict = json.loads(clean_response)
            d_llm = float(result_dict.get("heuristic_value", 0.5))
            
            # Bound the value mathematically just in case of hallucination
            d_llm = max(0.0, min(1.0, d_llm))
            
            # 5. Cache it for future iterations
            self.heuristic_buffer[state_key] = d_llm
            return d_llm
            
        except (json.JSONDecodeError, OutputParserException, ValueError) as e:
            print(f"LLM Parsing Error: {e}. Defaulting to neutral heuristic (0.5).")
            # If the LLM breaks, return a neutral 0.5 so the SARSA math doesn't crash
            return 0.5

# --- Quick Test Execution ---
if __name__ == "__main__":
    # Instantiate the engine
    engine = LLMHeuristicGenerator()
    
    # Simulate an incoming task from the Google Cluster Dataset (Cluster 7 - Huge Task)
    # Trying to schedule it on a weak, heavily loaded VM
    print("Testing bad allocation...")
    bad_val = engine.get_heuristic(task_mi=337500, task_type="huge", vm_cpu=0.5, vm_mem=0.03, vm_load=90)
    print(f"Returned Heuristic (Expected low): {bad_val}")
    
    # Simulate scheduling a tiny task on an idle VM
    print("\nTesting good allocation...")
    good_val = engine.get_heuristic(task_mi=15000, task_type="tiny", vm_cpu=1.0, vm_mem=1.0, vm_load=5)
    print(f"Returned Heuristic (Expected high): {good_val}")
