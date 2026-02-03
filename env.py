import os
import sys
import traci
import sumolib
import numpy as np
from gymnasium.spaces import Box, Discrete

class MultiAgentTrafficEnv:
    def __init__(self, sumo_cfg, use_gui=True):
        self.sumo_cfg = sumo_cfg
        self.use_gui = use_gui
        self.sumo_binary = sumolib.checkBinary('sumo-gui' if use_gui else 'sumo')
        
        # agent_ids will be set dynamically by train_multi.py
        self.agent_ids = []

        self.last_action = {} 
        
        # Priority weights based on vehicle capacity/impact for heterogeneous traffic
        self.vtype_weights = {
            "car": 1.0,
            "motorcycle": 0.5,
            "auto_rickshaw": 1.5,
            "truck": 3.0,
            "bus": 5.0, 
            "lcv": 2.0
        }

        # Observation: 6 types * 2 features (count, speed) = 12 inputs per agent
        # The training script handles the global_obs_dim for the Critic
        self.observation_spaces = {}
        self.action_spaces = {}

    def _get_state(self, agent_id):
        """Extracts 12-feature state vector for a specific junction."""
        controlled_lanes = traci.trafficlight.getControlledLanes(agent_id)
        controlled_edges = list(set([traci.lane.getEdgeID(l) for l in controlled_lanes]))
        
        state_vector = []
        target_vtypes = list(self.vtype_weights.keys())
        
        for vtype in target_vtypes:
            vehs_on_approach = []
            for edge in controlled_edges:
                all_vehs = traci.edge.getLastStepVehicleIDs(edge)
                for v in all_vehs:
                    if traci.vehicle.getTypeID(v) == vtype:
                        lane_pos = traci.vehicle.getLanePosition(v)
                        edge_len = traci.lane.getLength(traci.vehicle.getLaneID(v))
                        # 150m "Sensor Range" to focus on intersection-area traffic
                        if (edge_len - lane_pos) < 150: 
                            vehs_on_approach.append(v)

            count = len(vehs_on_approach)
            avg_speed = np.mean([traci.vehicle.getSpeed(v) for v in vehs_on_approach]) if vehs_on_approach else 0
            
            # Normalization (Max capacity ~100 vehs, Max speed ~25 m/s)
            state_vector.extend([min(count / 100.0, 1.0), min(avg_speed / 25.0, 1.0)])
            
        return np.array(state_vector, dtype=np.float32)

    def _calculate_reward(self, agent_id, current_action):
        """Calculates multi-objective reward: Efficiency, Fairness, and Stability."""
        total_penalty = 0
        
        # 1. Stability: Penalize phase changes to prevent 'Signal Panic'
        if current_action != self.last_action.get(agent_id, 0):
            total_penalty -= 50.0 
            
        controlled_lanes = traci.trafficlight.getControlledLanes(agent_id)
        
        for v_id in traci.vehicle.getIDList():
            if traci.vehicle.getLaneID(v_id) in controlled_lanes:
                v_type = traci.vehicle.getTypeID(v_id)
                delay = traci.vehicle.getTimeLoss(v_id) 
                
                # 2. Fairness: Capped Quadratic Waiting Time (Pressure)
                # min(raw_wait, 120) prevents mathematical explosion in the reward
                raw_wait = traci.vehicle.getWaitingTime(v_id)
                wait_time = min(raw_wait, 120) 
                pressure = (wait_time ** 2) 
                
                weight = self.vtype_weights.get(v_type, 1.0)
                total_penalty -= ((delay + pressure) * weight)
        
        self.last_action[agent_id] = current_action
        
        # 3. Normalization: Scales reward to ~(-1 to -50) for neural network stability
        return total_penalty / 10000.0

    def step(self, actions):
        """Executes one step in the environment."""
        for agent_id, action in actions.items():
            traci.trafficlight.setPhase(agent_id, action)
            
        # 5-second interval (10 steps of 0.5s)
        for _ in range(10): 
            traci.simulationStep()
            
        states = {a: self._get_state(a) for a in self.agent_ids}
        rewards = {a: self._calculate_reward(a, actions[a]) for a in self.agent_ids}
        
        # Termination at 1800s (30 minutes)
        done = traci.simulation.getTime() >= 1800 or traci.simulation.getMinExpectedNumber() <= 0
        dones = {a: done for a in self.agent_ids}
        
        return states, rewards, dones, {}

    def reset(self):
        """Resets simulation and dynamically detects junction action spaces."""
        if traci.isLoaded():
            traci.close()
            
        traci.start([
            self.sumo_binary, "-c", self.sumo_cfg, 
            "--waiting-time-memory", "1000",
            "--lateral-resolution", "0.2", # Enables sublane filtering
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--step-length", "0.5",
        ])

        # Dynamic phase detection for heterogeneous intersections
        for agent in self.agent_ids:
            num_phases = len(traci.trafficlight.getAllProgramLogics(agent)[0].phases)
            self.action_spaces[agent] = Discrete(num_phases)
            self.observation_spaces[agent] = Box(low=0, high=1, shape=(12,), dtype=np.float32)
            self.last_action[agent] = 0

        return {a: self._get_state(a) for a in self.agent_ids}

    def close(self):
        if traci.isLoaded():
            traci.close()