import torch
import numpy as np
from env import MultiAgentTrafficEnv
from model import MAPPOAgent

SUMO_CFG = r"E:\SUMO_Software\TestSim\Sample2\Sample1.sumocfg"
EPISODES = 150
STEPS_PER_EPISODE = 360 # 1800s / 5s

def train_multi_agent():
    env = MultiAgentTrafficEnv(SUMO_CFG, use_gui=False)
    # Define your Junction IDs from .net.xml
    junctions = ["clusterJ1", "clusterJ2_J4"] 
    env.agent_ids = junctions
    
    # 12 inputs per agent * 2 agents = 24 global_obs_dim
    global_obs_dim = 12 * len(junctions)
    
    agents = {
        j_id: MAPPOAgent(j_id, obs_dim=12, action_dim=8, global_obs_dim=global_obs_dim)
        for j_id in junctions
    }

    for episode in range(EPISODES):
        states = env.reset()
        episode_rewards = {j_id: 0 for j_id in junctions}

        for step in range(STEPS_PER_EPISODE):
            actions, log_probs = {}, {}
            
            # 1. Decentralized Action Selection
            for j_id in junctions:
                actions[j_id], log_probs[j_id] = agents[j_id].select_action(states[j_id])

            # 2. Environment Step
            next_states, rewards, dones, _ = env.step(actions)

            # 3. Centralized Training (CTDE)
            # Concatenate all states for the global critic
            global_state = np.concatenate([states[j] for j in junctions])
            global_state_t = torch.FloatTensor(global_state).unsqueeze(0)
            
            global_next_state = np.concatenate([next_states[j] for j in junctions])
            global_next_state_t = torch.FloatTensor(global_next_state).unsqueeze(0)

            for j_id in junctions:
                # Critic Update
                value = agents[j_id].critic(global_state_t)
                with torch.no_grad():
                    next_val = agents[j_id].critic(global_next_state_t)
                    target = rewards[j_id] + (0.99 * next_val * (1 - int(dones[j_id])))
                
                critic_loss = torch.nn.functional.mse_loss(value, target)
                agents[j_id].critic_optimizer.zero_grad()
                critic_loss.backward()
                agents[j_id].critic_optimizer.step()

                # Actor Update with Entropy Bonus
                curr_probs = agents[j_id].actor(torch.FloatTensor(states[j_id]).unsqueeze(0))
                dist = torch.distributions.Categorical(curr_probs)
                entropy = dist.entropy().mean()
                advantage = (target - value).detach()
                
                actor_loss = -(log_probs[j_id] * advantage) - (0.01 * entropy)
                agents[j_id].actor_optimizer.zero_grad()
                actor_loss.backward()
                agents[j_id].actor_optimizer.step()

                episode_rewards[j_id] += rewards[j_id]

            states = next_states
            if any(dones.values()): break

        print(f"Episode {episode} | Rewards: {episode_rewards}")
    env.close()

if __name__ == "__main__":
    train_multi_agent()