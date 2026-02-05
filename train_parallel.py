import torch
import numpy as np
import multiprocessing as mp
from core.env import MultiAgentTrafficEnv
from core.model import MAPPOAgent

# --- Configuration ---
NUM_WORKERS = 4  
SUMO_CFG = r"E:\SUMO_Software\TestSim\Sample2\Sample1.sumocfg"
EPISODES = 150
STEPS_PER_EPISODE = 360 
SAVE_FOLDER = "models_parallel_9j"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def worker_process(worker_id, child_pipe, junction_configs):
    env = MultiAgentTrafficEnv(SUMO_CFG, use_gui=False)
    env.agent_ids = list(junction_configs.keys())
    states = env.reset(label=f"worker_{worker_id}", port=8813 + worker_id)
    
    # Send the initial state to the manager so it can pick the first action
    child_pipe.send(states)

    while True:
        actions = child_pipe.recv()
        if actions is None: 
            env.close()
            break 
        
        next_states, rewards, dones, _ = env.step(actions)
        child_pipe.send((next_states, rewards, dones))
        states = next_states
        
        if any(dones.values()):
            states = env.reset(label=f"worker_{worker_id}", port=8813 + worker_id)

def train_parallel():
    junction_configs = {"J0": 6, "J11": 4, "J12": 4, "J13": 4, "J15": 4, "J17": 4, "J2": 4, "J8": 4, "clusterJ2_J4": 8}
    global_obs_dim = 12 * len(junction_configs)
    agents = {j: MAPPOAgent(j, 12, action_dim, global_obs_dim) for j, action_dim in junction_configs.items()}

    pipes, processes = [], []
    for i in range(NUM_WORKERS):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(target=worker_process, args=(i, child_conn, junction_configs))
        p.start()
        pipes.append(parent_conn)
        processes.append(p)

    # Initial states from all workers
    current_worker_states = [pipe.recv() for pipe in pipes]

    for episode in range(EPISODES):
        for step in range(STEPS_PER_EPISODE):
            for i, pipe in enumerate(pipes):
                # 1. Each agent picks an action based on its local observation
                worker_actions = {}
                worker_log_probs = {}
                for j_id in junction_configs:
                    action, log_prob = agents[j_id].select_action(current_worker_states[i][j_id])
                    worker_actions[j_id] = action
                    worker_log_probs[j_id] = log_prob

                # 2. Send actions to worker and receive results
                pipe.send(worker_actions)
                next_states, rewards, dones = pipe.recv()

                # 3. Centralized Training (CTDE)
                # Combine states into 108-dim vector
                global_s = np.concatenate([current_worker_states[i][j] for j in junction_configs.keys()])
                global_s_t = torch.FloatTensor(global_s).unsqueeze(0).to(device)
                
                global_next_s = np.concatenate([next_states[j] for j in junction_configs.keys()])
                global_next_s_t = torch.FloatTensor(global_next_s).unsqueeze(0).to(device)

                for j_id in junction_configs:
                    # Update Critic (Value estimation)
                    value = agents[j_id].critic(global_s_t)
                    with torch.no_grad():
                        next_val = agents[j_id].critic(global_next_s_t)
                        target = rewards[j_id] + (0.99 * next_val * (1 - int(dones[j_id])))
                    
                    critic_loss = torch.nn.functional.mse_loss(value, target)
                    agents[j_id].critic_optimizer.zero_grad()
                    critic_loss.backward()
                    agents[j_id].critic_optimizer.step()

                    # Update Actor (Policy Improvement)
                    curr_probs = agents[j_id].actor(torch.FloatTensor(current_worker_states[i][j_id]).unsqueeze(0).to(device))
                    dist = torch.distributions.Categorical(curr_probs)
                    entropy = dist.entropy().mean()
                    advantage = (target - value).detach()
                    actor_loss = -(worker_log_probs[j_id] * advantage) - (0.01 * entropy)
                    
                    agents[j_id].actor_optimizer.zero_grad()
                    actor_loss.backward()
                    agents[j_id].actor_optimizer.step()

                current_worker_states[i] = next_states

        if episode % 10 == 0:
            print(f"Episode {episode} Complete. Saving models...")
            for agent in agents.values(): agent.save_model(folder=SAVE_FOLDER)

    for pipe in pipes: pipe.send(None)
    for p in processes: p.join()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    train_parallel()