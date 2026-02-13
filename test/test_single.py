import torch
import traci
from core.env import MultiAgentTrafficEnv
from core.model import MAPPOAgent
from gymnasium.spaces import Discrete

# --- CONFIGURATION ---
SUMO_CFG = r"E:\SUMO_Software\TestSim\Sample2\Sample1.sumocfg"
MODEL_PATH = "best_single_agent.pth"
JUNCTION_ID = "clusterJ2_J4"

def test_trained_agent():
    # 1. Initialize Env with GUI on to visualize the 'intelligence'
    env = MultiAgentTrafficEnv(SUMO_CFG, use_gui=True)
    env.agent_ids = [JUNCTION_ID]
    env.action_spaces = {JUNCTION_ID: Discrete(8)}

    # 2. Initialize Agent and Load Weights
    agent = MAPPOAgent(JUNCTION_ID, obs_dim=12, action_dim=8, global_obs_dim=12)
    
    try:
        agent.actor.load_state_dict(torch.load(MODEL_PATH))
        agent.actor.eval() # Set to evaluation mode (turns off dropout/batchnorm)
        print(f">>> Successfully loaded {MODEL_PATH}")
    except FileNotFoundError:
        print("Error: Model file not found. Check the path.")
        return

    # 3. Run Simulation
    states = env.reset()
    total_reward = 0
    step_count = 0
    
    print(f"--- Running Test Simulation for {JUNCTION_ID} ---")

    while True:
        # AI selects the best action based on the trained policy (no randomness)
        with torch.no_grad():
            state_t = torch.FloatTensor(states[JUNCTION_ID]).unsqueeze(0)
            probs = agent.actor(state_t)
            action = torch.argmax(probs).item()
            print(f"AI Step: {step_count} | Selected Phase: {action} | Probs: {probs.numpy()}") # Add this line

        # Step the environment
        next_states, rewards, dones, _ = env.step({JUNCTION_ID: action})
        
        total_reward += rewards[JUNCTION_ID]
        states = next_states
        step_count += 1

        if any(dones.values()):
            break

    print("-" * 30)
    print(f"Test Finished after {step_count} steps.")
    print(f"Final AI Cumulative Reward: {total_reward:.2f}")
    print("-" * 30)
    
    env.close()

if __name__ == "__main__":
    test_trained_agent()