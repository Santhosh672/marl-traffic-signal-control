import torch
import numpy as np
import multiprocessing as mp
import os
from core.env import MultiAgentTrafficEnv
from core.model import MAPPOAgent

# --- Configuration & Path Hygiene ---
NUM_WORKERS = 4  
# Use absolute paths or paths relative to the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUMO_CFG = r"E:\SUMO_Software\TestSim\Sample2\Sample1.sumocfg"
SAVE_FOLDER = os.path.join(BASE_DIR, "output", "model")
EPISODES = 150
STEPS_PER_EPISODE = 360 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def worker_process(worker_id, child_pipe, junction_configs):
    """Sub-process running a dedicated SUMO instance."""
    env = MultiAgentTrafficEnv(SUMO_CFG, use_gui=False)
    env.agent_ids = list(junction_configs.keys())
    
    # Each worker needs a unique label and port to run in parallel
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
    # 9-junction mapping from your network
    junction_configs = {
        "J0": 6, "J11": 4, "J12": 4, "J13": 4, "J15": 4, 
        "J17": 4, "J2": 4, "J8": 4, "clusterJ2_J4": 8
    }
    
    # 108 global features = 9 agents * 12 local features
    global_obs_dim = 12 * len(junction_configs)
    agents = {j: MAPPOAgent(j, 12, action_dim, global_obs_dim) for j, action_dim in junction_configs.items()}

    # Initialize pipes and processes
    pipes, processes = [], []
    for i in range(NUM_WORKERS):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(target=worker_process, args=(i, child_conn, junction_configs))
        p.start()
        pipes.append(parent_conn)
        processes.append(p)

    # Initial states from all workers to bootstrap the loop
    current_worker_states = [pipe.recv() for pipe in pipes]

    print(f"Parallel Training Started: {NUM_WORKERS} workers active.")

    for episode in range(EPISODES):
        for step in range(STEPS_PER_EPISODE):
            # --- BROADCAST PHASE ---
            # Collect actions for all workers and send them simultaneously
            worker_log_probs_batch = []
            worker_actions_batch = []

            for i in range(NUM_WORKERS):
                actions = {}
                log_probs = {}
                for j_id in junction_configs:
                    action, log_prob = agents[j_id].select_action(current_worker_states[i][j_id])
                    actions[j_id] = action
                    log_probs[j_id] = log_prob
                
                worker_actions_batch.append(actions)
                worker_log_probs_batch.append(log_probs)
                pipes[i].send(actions) # Non-blocking send

            # --- COLLECT PHASE ---
            # Now wait for all workers to finish their 5-second simulation step
            worker_results = [pipe.recv() for pipe in pipes] 

            # --- UPDATE PHASE (Centralized Training) ---
            for i, (next_states, rewards, dones) in enumerate(worker_results):
                # Concatenate all 9 agent states for the 108-dim Critic
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
                    curr_obs_t = torch.FloatTensor(current_worker_states[i][j_id]).unsqueeze(0).to(device)
                    curr_probs = agents[j_id].actor(curr_obs_t)
                    dist = torch.distributions.Categorical(curr_probs)
                    entropy = dist.entropy().mean()
                    advantage = (target - value).detach()
                    
                    # Log-prob from the specific worker action
                    actor_loss = -(worker_log_probs_batch[i][j_id] * advantage) - (0.01 * entropy)
                    
                    agents[j_id].actor_optimizer.zero_grad()
                    actor_loss.backward()
                    agents[j_id].actor_optimizer.step()

                # Update local state for the next step
                current_worker_states[i] = next_states

        if episode % 10 == 0:
            print(f"Episode {episode} Complete. Saving models to {SAVE_FOLDER}...")
            for agent in agents.values(): 
                agent.save_model(folder=SAVE_FOLDER) # Directed to output/model

    # Cleanup
    for pipe in pipes: pipe.send(None)
    for p in processes: p.join()

if __name__ == "__main__":
    # Required for Windows multiprocessing safety
    mp.set_start_method('spawn', force=True)
    train_parallel()