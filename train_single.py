import torch
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.spaces import Discrete
from env import MultiAgentTrafficEnv
from model import MAPPOAgent

# --- CONFIGURATION FROM YOUR PROJECT FILES ---
# Path updated to your directory structure
SUMO_CFG = r"E:\SUMO_Software\TestSim\Sample2\Sample1.sumocfg" 
EPISODES = 50
STEPS_PER_EPISODE = 360 # (3600s sim / 5s intervals)

def train_single_agent():
    log_file = "training_log.txt"
    
    # 1. Initialize Environment
    # use_gui=True lets you watch the 6 vehicle types interact
    env = MultiAgentTrafficEnv(SUMO_CFG, use_gui=False)
    
    # 2. Configure for your specific Junction
    agent_id = "clusterJ2_J4"
    env.agent_ids = [agent_id]
    env.action_spaces = {agent_id: Discrete(8)} # Matches the 8 phases in your .net file

    # 3. Initialize Agent
    # obs_dim=12 (6 types * [count, speed]), action_dim=8, global_obs_dim=12
    agent = MAPPOAgent(agent_id, obs_dim=12, action_dim=8, global_obs_dim=12)
    
    rewards_history = []
    best_so_far = 0

    print(f"--- Starting Training for Junction: {agent_id} ---")

    for episode in range(EPISODES):
        states = env.reset()
        episode_reward = 0

        print(f"Started training episode: {episode}")
        
        for step in range(STEPS_PER_EPISODE):
            # AI selects a phase (0-7)
            action, log_prob = agent.select_action(states[agent_id])
            
            # Environment steps forward 5 seconds
            next_states, rewards, dones, _ = env.step({agent_id: action})
            
            # Convert observations to PyTorch Tensors
            state_t = torch.FloatTensor(states[agent_id]).unsqueeze(0)
            next_state_t = torch.FloatTensor(next_states[agent_id]).unsqueeze(0)
            
            # --- STEP 1: CRITIC UPDATE (Value Estimation) ---
            # Predict the value of the current state
            value = agent.critic(state_t)
            
            # Calculate Target using Bellman Equation
            with torch.no_grad():
                next_value = agent.critic(next_state_t)
                td_target = rewards[agent_id] + (0.99 * next_value * (1 - int(dones[agent_id])))
            
            # Calculate Critic Loss (MSE)
            critic_loss = torch.nn.functional.mse_loss(value, td_target)
            
            # Backpropagate Critic
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()

            # --- STEP 2: ACTOR UPDATE (Policy Strategy) ---
            # Detach the advantage to prevent 'backward through graph twice' error
            # This follows the CTDE training logic
            advantage = (td_target - value).detach()
            actor_loss = -(log_prob * advantage)
            
            # Backpropagate Actor
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

            episode_reward += rewards[agent_id]
            states = next_states
            
            if dones[agent_id]:
                break
        
        if episode > 0:
            best_so_far = max(rewards_history)
    
        # Log to file
        with open(log_file, "a") as f:
            f.write(f"{episode},{episode_reward:.2f},{best_so_far:.2f}\n")
        
        # Print only once per episode to keep terminal clean
        print(f"Episode {episode} Complete | Reward: {episode_reward:.2f} | Best: {best_so_far:.2f}")

        rewards_history.append(episode_reward)
        print(f"Episode {episode} | Cumulative Reward: {episode_reward:.2f}")

        # Save the best model based on weighted delay penalty
        if episode > 0 and rewards_history[-1] > max(rewards_history[:-1]):
            torch.save(agent.actor.state_dict(), "best_single_agent.pth")
            print(">>> New Best Model Saved!")

    env.close()
    return rewards_history

if __name__ == "__main__":
    try:
        history = train_single_agent()
        
        # Plotting the Learning Curve for your IEEE Paper
        plt.figure(figsize=(10, 5))
        plt.plot(history, color='blue', label='MAPPO Agent')
        plt.title("Learning Curve: Traffic Delay Optimization")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward (Negative Delay)")
        plt.grid(True)
        plt.savefig("learning_curve.png")
        plt.show()
    except Exception as e:
        print(f"Training interrupted: {e}")