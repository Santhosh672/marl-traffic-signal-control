import os
import torch
import pandas as pd
import traci
from core.env import MultiAgentTrafficEnv
from core.model import MAPPOAgent

# --- Configuration ---
SUMO_CFG = r"E:\SUMO_Software\TestSim\Sample2\Sample1.sumocfg"
MODEL_DIR = "output/model"
LOG_DIR = "output/log/junction_phases"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

def run_and_log_phases():
    # 1. Temporary Env to discover CORRECT action dimensions (Green-Only)
    # This prevents the size mismatch error by matching the model to the filtered phases
    temp_env = MultiAgentTrafficEnv(SUMO_CFG, use_gui=False)
    junction_ids = ["J0", "J11", "J12", "J13", "J15", "J17", "J2", "J8", "clusterJ2_J4"]
    temp_env.agent_ids = junction_ids
    temp_env.reset() 
    
    # Map IDs to the actual number of Green phases found in your XML
    junction_configs = {j: temp_env.action_spaces[j].n for j in junction_ids}
    temp_env.close()
    
    print(f"Action Spaces Discovered: {junction_configs}")

    # 2. Initialize Main Env with GUI
    env = MultiAgentTrafficEnv(SUMO_CFG, use_gui=True)
    env.agent_ids = junction_ids
    global_obs_dim = 12 * len(env.agent_ids)
    
    # 3. Load Agents using discovered dimensions
    agents = {}
    logs = {j_id: [] for j_id in env.agent_ids}
    
    print(">>> Loading Models and Starting Logger...")
    for j_id, action_dim in junction_configs.items():
        # Initialize agent with the correct dimension (e.g. 3 instead of 6)
        agent = MAPPOAgent(j_id, 12, action_dim, global_obs_dim)
        path = os.path.join(MODEL_DIR, f"actor_{j_id}.pth")
        if os.path.exists(path):
            agent.actor.load_state_dict(torch.load(path, map_location=device))
            agent.actor.eval()
            agents[j_id] = agent
        else:
            print(f"Warning: Model for {j_id} not found at {path}")

    # 4. Simulation Loop
    states = env.reset()
    step_count = 0

    try:
        while step_count < 360:  # Log for 30 minutes of simulation
            actions = {}
            for j_id, agent in agents.items():
                with torch.no_grad():
                    state_t = torch.FloatTensor(states[j_id]).unsqueeze(0).to(device)
                    probs = agent.actor(state_t)
                    action = torch.argmax(probs).item()
                    actions[j_id] = action
                    
                    # Capture the light state string (e.g., "GGrr")
                    light_string = traci.trafficlight.getRedYellowGreenState(j_id)
                    
                    logs[j_id].append({
                        "Step": step_count,
                        "Time": traci.simulation.getTime(),
                        "Phase_Index": action,
                        "Light_State": light_string
                    })

            # Execute actions in SUMO (env.step maps these to XML indices)
            next_states, _, dones, _ = env.step(actions)
            states = next_states
            step_count += 1
            
            if any(dones.values()): break

    finally:
        # 5. Save individual CSVs for analysis
        for j_id, data in logs.items():
            if data:
                df = pd.DataFrame(data)
                log_path = os.path.join(LOG_DIR, f"{j_id}_actions.csv")
                df.to_csv(log_path, index=False)
                print(f"Logged {len(df)} steps for {j_id} to {log_path}")
        
        env.close()

if __name__ == "__main__":
    run_and_log_phases()