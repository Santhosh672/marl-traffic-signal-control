import torch
import numpy as np
from env import MultiAgentTrafficEnv
from model import MAPPOAgent

# Configuration
SUMO_CFG = r"E:\SUMO_Software\TestSim\Sample2\Sample1.sumocfg"
EPISODES = 150
STEPS_PER_EPISODE = 360 # 1800s / 5s intervals

def train_multi_agent():
    env = MultiAgentTrafficEnv(SUMO_CFG, use_gui=False)
    
    # 1. Map Junction IDs to their types based on your .net.xml
    # Note: J0 is 3-way (6 actions), clusterJ2_J4 is 4-way (12 actions)
    junction_configs = {
        "J0": 6,
        "clusterJ2_J4": 12
    }
    env.agent_ids = list(junction_configs.keys())
    
    # Global state is the sum of all local observations (12 * 2 = 24)
    global_obs_dim = 12 * len(env.agent_ids)
    
    # 2. Initialize Agents with matching dimensions
    agents = {
        j_id: MAPPOAgent(j_id, 12, action_dim, global_obs_dim)
        for j_id, action_dim in junction_configs.items()
    }

    for episode in range(EPISODES):
        states = env.reset()
        episode_rewards = {j_id: 0 for j_id in env.agent_ids}

        for step in range(STEPS_PER_EPISODE):
            actions, log_probs = {}, {}
            
            # DECENTRALIZED EXECUTION: Agents act based only on local obs
            for j_id in env.agent_ids:
                actions[j_id], log_probs[j_id] = agents[j_id].select_action(states[j_id])

            next_states, rewards, dones, _ = env.step(actions)

            # CENTRALIZED TRAINING: Critic sees the entire map state
            global_state = np.concatenate([states[j] for j in env.agent_ids])
            global_state_t = torch.FloatTensor(global_state).unsqueeze(0)
            
            global_next_state = np.concatenate([next_states[j] for j in env.agent_ids])
            global_next_state_t = torch.FloatTensor(global_next_state).unsqueeze(0)

            for j_id in env.agent_ids:
                # --- CRITIC UPDATE ---
                value = agents[j_id].critic(global_state_t)
                with torch.no_grad():
                    next_val = agents[j_id].critic(global_next_state_t)
                    # Reward normalization is already handled in env.py
                    target = rewards[j_id] + (0.99 * next_val * (1 - int(dones[j_id])))
                
                critic_loss = torch.nn.functional.mse_loss(value, target)
                agents[j_id].critic_optimizer.zero_grad()
                critic_loss.backward()
                agents[j_id].critic_optimizer.step()

                # --- ACTOR UPDATE (With Entropy Bonus) ---
                curr_probs = agents[j_id].actor(torch.FloatTensor(states[j_id]).unsqueeze(0))
                dist = torch.distributions.Categorical(curr_probs)
                entropy = dist.entropy().mean()
                advantage = (target - value).detach()
                
                # Entropy bonus (0.01) prevents sticking to one phase
                actor_loss = -(log_probs[j_id] * advantage) - (0.01 * entropy)
                
                agents[j_id].actor_optimizer.zero_grad()
                actor_loss.backward()
                agents[j_id].actor_optimizer.step()

                episode_rewards[j_id] += rewards[j_id]

            states = next_states
            if any(dones.values()): break

        print(f"Episode {episode} | Avg Reward: {np.mean(list(episode_rewards.values())):.2f}")
    
    # Save models after training
    for j_id, agent in agents.items():
        torch.save(agent.actor.state_dict(), f"actor_{j_id}.pth")

    env.close()

if __name__ == "__main__":
    train_multi_agent()