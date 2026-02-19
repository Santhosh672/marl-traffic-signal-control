import torch
import numpy as np
import multiprocessing as mp
import os
import pandas as pd
from core.env import MultiAgentTrafficEnv
from core.model import MAPPOAgent

# --- Configuration ---
NUM_WORKERS = 4  
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUMO_CFG = r"E:\SUMO_Software\TestSim\Sample2\Sample1.sumocfg"
SAVE_FOLDER = os.path.join(BASE_DIR, "output", "model")
EPISODES = 150
STEPS_PER_EPISODE = 360 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Check for CUDA ---
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

def worker_process(worker_id, child_pipe, junction_ids):
    """Sub-process running a dedicated SUMO instance."""
    env = MultiAgentTrafficEnv(SUMO_CFG, use_gui=False)
    env.agent_ids = junction_ids
    # reset() here will trigger the Green-Only filtering in env.py
    states = env.reset(label=f"worker_{worker_id}", port=8813 + worker_id)
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
    # 1. Tracker Initialization (Prevents NameError)
    rewards_history = []
    junction_ids = ["J0", "J11", "J12", "J13", "J15", "J17", "J2", "J8", "clusterJ2_J4"]
    
    # 2. Dynamic Action Space Discovery (Prevents IndexError)
    # We run a brief temp env to see how many Green phases env.py found for each junction
    temp_env = MultiAgentTrafficEnv(SUMO_CFG, use_gui=False)
    temp_env.agent_ids = junction_ids
    temp_env.reset() 
    junction_configs = {j: temp_env.action_spaces[j].n for j in junction_ids}
    temp_env.close()
    
    print(f"Action Spaces Discovered: {junction_configs}")

    # 3. Agent Initialization
    global_obs_dim = 12 * len(junction_ids)
    agents = {j: MAPPOAgent(j, 12, junction_configs[j], global_obs_dim) for j in junction_ids}

    # 4. Process Setup
    pipes, processes = [], []
    for i in range(NUM_WORKERS):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(target=worker_process, args=(i, child_conn, junction_ids))
        p.start()
        pipes.append(parent_conn)
        processes.append(p)

    current_worker_states = [pipe.recv() for pipe in pipes]
    print(f"Parallel Training Started: {NUM_WORKERS} workers active.")

    # 5. Training Loop
    for episode in range(EPISODES):
        ep_rewards = [] 
        for step in range(STEPS_PER_EPISODE):
            # BROADCAST
            worker_log_probs_batch = []
            for i in range(NUM_WORKERS):
                actions, log_probs = {}, {}
                for j_id in junction_ids:
                    action, log_prob = agents[j_id].select_action(current_worker_states[i][j_id])
                    actions[j_id], log_probs[j_id] = action, log_prob
                
                worker_log_probs_batch.append(log_probs)
                pipes[i].send(actions)

            # COLLECT
            worker_results = [pipe.recv() for pipe in pipes] 

            # UPDATE
            for i, (next_states, rewards, dones) in enumerate(worker_results):
                ep_rewards.append(np.mean(list(rewards.values())))
                
                global_s = np.concatenate([current_worker_states[i][j] for j in junction_ids])
                global_s_t = torch.FloatTensor(global_s).unsqueeze(0).to(device)
                global_next_s = np.concatenate([next_states[j] for j in junction_ids])
                global_next_s_t = torch.FloatTensor(global_next_s).unsqueeze(0).to(device)

                for j_id in junction_ids:
                    value = agents[j_id].critic(global_s_t)
                    with torch.no_grad():
                        next_val = agents[j_id].critic(global_next_s_t)
                        target = rewards[j_id] + (0.99 * next_val * (1 - int(dones[j_id])))
                    
                    critic_loss = torch.nn.functional.mse_loss(value, target)
                    agents[j_id].critic_optimizer.zero_grad()
                    critic_loss.backward()
                    agents[j_id].critic_optimizer.step()

                    curr_obs_t = torch.FloatTensor(current_worker_states[i][j_id]).unsqueeze(0).to(device)
                    curr_probs = agents[j_id].actor(curr_obs_t)
                    dist = torch.distributions.Categorical(curr_probs)
                    entropy = dist.entropy().mean()
                    advantage = (target - value).detach()
                    
                    actor_loss = -(worker_log_probs_batch[i][j_id] * advantage) - (0.01 * entropy)
                    agents[j_id].actor_optimizer.zero_grad()
                    actor_loss.backward()
                    agents[j_id].actor_optimizer.step()

                current_worker_states[i] = next_states

        # Log Episode
        avg_reward = np.mean(ep_rewards) if ep_rewards else 0
        rewards_history.append(avg_reward)

        if episode % 10 == 0:
            print(f"Episode {episode}")
            for agent in agents.values(): 
                agent.save_model(folder=SAVE_FOLDER)

    # 6. Final Report Generation
    summary_path = os.path.join(BASE_DIR, "output", "stats", "training_summary.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    pd.DataFrame({"Episode": range(len(rewards_history)), "Avg_Reward": rewards_history}).to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

    # 7. Cleanup
    for pipe in pipes: pipe.send(None)
    for p in processes: p.join()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    try:
        train_parallel()
        print("Training complete. Laptop shutting down in 60s...")
    except Exception as e:
        print(f"Training crashed: {e}. Shutting down to save power.")
    finally:
        os.system("shutdown /s /f /t 60")