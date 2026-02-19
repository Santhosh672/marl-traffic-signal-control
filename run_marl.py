import os
import torch
import numpy as np
from core.env import MultiAgentTrafficEnv
from core.model import MAPPOAgent

# --- Configuration ---
SUMO_CFG = r"E:\SUMO_Software\TestSim\Sample2\Sample1.sumocfg"
MODEL_DIR = "output/model"  # Where your .pth files are stored
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_trained_marl():
    # 1. Temporary Env to discover CORRECT action dimensions (Green-Only)
    # We use use_gui=False here just for discovery to avoid flickering windows
    temp_env = MultiAgentTrafficEnv(SUMO_CFG, use_gui=False)
    junction_ids = ["J0", "J11", "J12", "J13", "J15", "J17", "J2", "J8", "clusterJ2_J4"]
    temp_env.agent_ids = junction_ids
    temp_env.reset() # Triggers the filtering logic in env.py
    
    # Map IDs to the actual number of Green phases found
    junction_configs = {j: temp_env.action_spaces[j].n for j in junction_ids}
    temp_env.close()
    
    print(f"Action Spaces Discovered: {junction_configs}")

    # 2. Initialize Main Env with GUI
    env = MultiAgentTrafficEnv(SUMO_CFG, use_gui=True)
    env.agent_ids = junction_ids
    global_obs_dim = 12 * len(env.agent_ids)
    
    # 3. Initialize and Load Agents using discovered dimensions
    agents = {}
    print(">>> Loading Trained Models...")
    for j_id, action_dim in junction_configs.items():
        # Initialize agent with the CORRECT dimension (e.g. 3 instead of 6)
        agent = MAPPOAgent(j_id, 12, action_dim, global_obs_dim)
        actor_path = os.path.join(MODEL_DIR, f"actor_{j_id}.pth")
        
        if os.path.exists(actor_path):
            agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
            agent.actor.eval()
            agents[j_id] = agent
        else:
            print(f"Warning: Model for {j_id} not found at {actor_path}")

    # 4. Run Simulation Loop
    states = env.reset()
    total_rewards = {j: 0 for j in env.agent_ids}
    step_count = 0

    print(f"--- Running Trained MARL Simulation ---")

    try:
        while True:
            actions = {}
            for j_id, agent in agents.items():
                with torch.no_grad():
                    state_t = torch.FloatTensor(states[j_id]).unsqueeze(0).to(device)
                    probs = agent.actor(state_t)
                    actions[j_id] = torch.argmax(probs).item()

            # Execute actions in SUMO (env.step maps these to XML indices)
            next_states, rewards, dones, _ = env.step(actions)
            
            for j_id in rewards:
                total_rewards[j_id] += rewards[j_id]
            
            states = next_states
            step_count += 1

            if step_count % 10 == 0:
                avg_reward = np.mean(list(total_rewards.values()))
                print(f"Step: {step_count} | Avg Network Reward: {avg_reward:.4f}")

            if any(dones.values()):
                break
    except Exception as e:
        print(f"Simulation ended: {e}")

    print("-" * 30)
    print(f"Test Finished after {step_count} steps.")
    env.close()

if __name__ == "__main__":
    run_trained_marl()