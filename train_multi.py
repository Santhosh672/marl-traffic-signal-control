import torch
import numpy as np
from core.env import MultiAgentTrafficEnv
from core.model import MAPPOAgent

# Configuration
SUMO_CFG = r"E:\SUMO_Software\TestSim\Sample2\Sample1.sumocfg"
EPISODES = 150
STEPS_PER_EPISODE = 360 # 1800s / 5s intervals

def train_multi_agent():
    env = MultiAgentTrafficEnv(SUMO_CFG, use_gui=False)
    
    # 1. Map all 9 Junction IDs to their phase counts from Sample2.net.xml
    junction_configs = {
        "J0": 6,              # T-Intersection
        "J11": 4,             # T-Intersection
        "J12": 4,             # T-Intersection
        "J13": 4,             # 4-Way Intersection
        "J15": 4,             # T-Intersection
        "J17": 4,             # T-Intersections
        "J2": 4,              # T-Intersection
        "J8": 4,              # T-Intersection
        "clusterJ2_J4": 8     # 4-Way Intersection
    }
    env.agent_ids = list(junction_configs.keys())
    
    # 2. Update Global Observation Dimension: 12 features * 9 agents = 108
    global_obs_dim = 12 * len(env.agent_ids)
    
    # 3. Initialize all 9 Agents
    agents = {
        j_id: MAPPOAgent(j_id, 12, action_dim, global_obs_dim)
        for j_id, action_dim in junction_configs.items()
    }

    print(f"Starting Multi-Agent Training with {len(agents)} agents...")

    for episode in range(EPISODES):
        states = env.reset()
        episode_rewards = {j_id: 0 for j_id in env.agent_ids}

        for step in range(STEPS_PER_EPISODE):
            actions, log_probs = {}, {}
            
            # Decentralized Execution: Each agent picks an action based on its local 12-dim state
            for j_id in env.agent_ids:
                actions[j_id], log_probs[j_id] = agents[j_id].select_action(states[j_id])

            next_states, rewards, dones, _ = env.step(actions)

            # Centralized Training: The Global State (108-dim) is used for coordination
            global_state = np.concatenate([states[j] for j in env.agent_ids])
            global_state_t = torch.FloatTensor(global_state).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            
            global_next_state = np.concatenate([next_states[j] for j in env.agent_ids])
            global_next_state_t = torch.FloatTensor(global_next_state).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            for j_id in env.agent_ids:
                # --- CRITIC UPDATE (Value Estimation) ---
                value = agents[j_id].critic(global_state_t)
                with torch.no_grad():
                    next_val = agents[j_id].critic(global_next_state_t)
                    target = rewards[j_id] + (0.99 * next_val * (1 - int(dones[j_id])))
                
                critic_loss = torch.nn.functional.mse_loss(value, target)
                agents[j_id].critic_optimizer.zero_grad()
                critic_loss.backward()
                agents[j_id].critic_optimizer.step()

                # --- ACTOR UPDATE (Policy Improvement with Entropy Bonus) ---
                curr_probs = agents[j_id].actor(torch.FloatTensor(states[j_id]).unsqueeze(0).to(global_state_t.device))
                dist = torch.distributions.Categorical(curr_probs)
                entropy = dist.entropy().mean()
                advantage = (target - value).detach()
                
                # The entropy bonus (0.01) encourages exploration to prevent policy collapse
                actor_loss = -(log_probs[j_id] * advantage) - (0.01 * entropy)
                
                agents[j_id].actor_optimizer.zero_grad()
                actor_loss.backward()
                agents[j_id].actor_optimizer.step()

                episode_rewards[j_id] += rewards[j_id]

            states = next_states
            if any(dones.values()): break

        avg_reward = np.mean(list(episode_rewards.values()))
        print(f"Episode {episode} | Avg Network Reward: {avg_reward:.2f}")
    
    # Save all 9 models after training
    for j_id, agent in agents.items():
        agent.save_model(folder="models_9_junctions")

    env.close()

if __name__ == "__main__":
    train_multi_agent()