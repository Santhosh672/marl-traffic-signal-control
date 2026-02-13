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
    # 1. Initialize Env with GUI
    env = MultiAgentTrafficEnv(SUMO_CFG, use_gui=True)
    
    # 2. Define Junctions
    junction_configs = {
        "J0": 6, "J11": 4, "J12": 4, "J13": 4, "J15": 4, 
        "J17": 4, "J2": 4, "J8": 4, "clusterJ2_J4": 8
    }
    env.agent_ids = list(junction_configs.keys())
    global_obs_dim = 12 * len(env.agent_ids)
    
    # 3. Load Agents
    agents = {}
    logs = {j_id: [] for j_id in env.agent_ids}
    
    print(">>> Loading Models and Starting Logger...")
    for j_id, action_dim in junction_configs.items():
        agent = MAPPOAgent(j_id, 12, action_dim, global_obs_dim)
        path = os.path.join(MODEL_DIR, f"actor_{j_id}.pth")
        if os.path.exists(path):
            agent.actor.load_state_dict(torch.load(path, map_location=device))
            agent.actor.eval()
            agents[j_id] = agent

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

            next_states, _, dones, _ = env.step(actions)
            states = next_states
            step_count += 1
            
            if any(dones.values()): break

    finally:
        # 5. Save individual CSVs for analysis
        for j_id, data in logs.items():
            df = pd.DataFrame(data)
            log_path = os.path.join(LOG_DIR, f"{j_id}_actions.csv")
            df.to_csv(log_path, index=False)
            print(f"Logged {len(df)} steps for {j_id} to {log_path}")
        
        env.close()

if __name__ == "__main__":
    run_and_log_phases()