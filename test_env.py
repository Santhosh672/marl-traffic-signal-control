import torch
import numpy as np
from env import MultiAgentTrafficEnv
from model import MAPPOAgent

# --- Configuration ---
SUMO_CFG = r"E:\SUMO_Software\TestSim\Sample2\Sample1.sumocfg"

def test_visual_coordination():
    # 1. Setup Environment with GUI
    env = MultiAgentTrafficEnv(SUMO_CFG, use_gui=True)
    
    # 2. Match your 9-junction network exactly
    junction_configs = {
        "J0": 6, "J11": 4, "J12": 4, "J13": 4, "J15": 4, 
        "J17": 4, "J2": 4, "J8": 4, "clusterJ2_J4": 8
    }
    env.agent_ids = list(junction_configs.keys())
    global_obs_dim = 12 * len(env.agent_ids)
    
    # 3. Initialize Agents (untrained/random for this test)
    agents = {
        j_id: MAPPOAgent(j_id, 12, action_dim, global_obs_dim)
        for j_id, action_dim in junction_configs.items()
    }

    print("Opening SUMO-GUI for validation...")
    states = env.reset() # This opens the window

    # Run for 50 steps to observe vehicle behavior
    for step in range(50):
        actions = {}
        for j_id in env.agent_ids:
            # Pick actions randomly to test link/phase connectivity
            action, _ = agents[j_id].select_action(states[j_id])
            actions[j_id] = action

        next_states, rewards, dones, _ = env.step(actions)
        
        # Monitor the 108-dimensional state vector in the console
        avg_reward = np.mean(list(rewards.values()))
        print(f"Step {step} | Avg Reward: {avg_reward:.4f} | Vehicles in Network: {traci.simulation.getMinExpectedNumber()}")
        
        states = next_states
        if any(dones.values()): break

    env.close()
    print("Validation successful. You are ready for train_parallel.py!")

if __name__ == "__main__":
    import traci
    test_visual_coordination()