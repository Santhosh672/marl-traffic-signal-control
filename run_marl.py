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
    # 1. Initialize Env with GUI
    env = MultiAgentTrafficEnv(SUMO_CFG, use_gui=True)
    
    # 2. Map Junction IDs to their phase counts from Sample2.net.xml
    junction_configs = {
        "J0": 6, "J11": 4, "J12": 4, "J13": 4, "J15": 4, 
        "J17": 4, "J2": 4, "J8": 4, "clusterJ2_J4": 8
    }
    env.agent_ids = list(junction_configs.keys())
    global_obs_dim = 12 * len(env.agent_ids) # 108 features total
    
    # 3. Initialize and Load Agents
    agents = {}
    print(">>> Loading Trained Models...")
    for j_id, action_dim in junction_configs.items():
        agent = MAPPOAgent(j_id, 12, action_dim, global_obs_dim)
        actor_path = os.path.join(MODEL_DIR, f"actor_{j_id}.pth")
        
        if os.path.exists(actor_path):
            agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
            agent.actor.eval() # Set to evaluation mode
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
                # Inference: Pick the best action (argmax) instead of sampling
                with torch.no_grad():
                    state_t = torch.FloatTensor(states[j_id]).unsqueeze(0).to(device)
                    probs = agent.actor(state_t)
                    actions[j_id] = torch.argmax(probs).item()

            # Execute actions in SUMO
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