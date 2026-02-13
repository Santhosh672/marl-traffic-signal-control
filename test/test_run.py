from core.env import MultiAgentTrafficEnv

# 1. Initialize the environment with your config
sumo_cfg_path = r"E:\SUMO_Software\TestSim\Sample2\Sample1.sumocfg"
env = MultiAgentTrafficEnv(sumo_cfg=sumo_cfg_path, use_gui=True)

# 2. Reset the environment
obs = env.reset()
print("Initial Observations:", obs)

# 3. Take 5 random steps
for i in range(5):
    # Pick a random phase index (0-11) for your intersection
    actions = {"clusterJ2_J4": 0} 
    states, rewards, dones, info = env.step(actions)
    print(f"Step {i} Reward:", rewards)

env.close()