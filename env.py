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
        
        # Match these to your .net.xml junction ID
        self.agent_ids = ["clusterJ2_J4"] 
        
        # IDs must match your vtypes.add.xml exactly
        self.vtype_weights = {
            "car": 1.0,
            "motorcycle": 0.5,
            "auto_rickshaw": 1.5,
            "truck": 3.0,
            "bus": 5.0, # High weight for high-capacity
            "lcv": 2.0
        }

        # Observation: 6 types * 2 features (count, speed) = 12 inputs per agent
        self.observation_spaces = {
            agent: Box(low=0, high=1, shape=(12,), dtype=np.float32) 
            for agent in self.agent_ids
        }

        # Action: Set this to the number of phases in your tlLogic
        self.action_spaces = {
            agent: Discrete(12) for agent in self.agent_ids
        }

    def _get_state(self, agent_id):
        # Edge-based filtering to handle "no lane discipline" sublane behavior
        controlled_lanes = traci.trafficlight.getControlledLanes(agent_id)
        controlled_edges = list(set([traci.lane.getEdgeID(l) for l in controlled_lanes]))
        
        state_vector = []
        target_vtypes = list(self.vtype_weights.keys())
        
        for vtype in target_vtypes:
            vehs_on_approach = []
            for edge in controlled_edges:
                all_vehs = traci.edge.getLastStepVehicleIDs(edge)
                for v in all_vehs:
                    # Filter by type and ensure they are moving TOWARD the light
                    if traci.vehicle.getTypeID(v) == vtype:
                        lane_pos = traci.vehicle.getLanePosition(v)
                        edge_len = traci.lane.getLength(traci.vehicle.getLaneID(v))
                        if (edge_len - lane_pos) < 150: # 150m "Sensor Range"
                            vehs_on_approach.append(v)

            count = len(vehs_on_approach)
            avg_speed = np.mean([traci.vehicle.getSpeed(v) for v in vehs_on_approach]) if vehs_on_approach else 0
            
            # Normalization for Indian densities
            state_vector.extend([min(count / 100.0, 1.0), min(avg_speed / 25.0, 1.0)])
            
        return np.array(state_vector, dtype=np.float32)

    def _calculate_reward(self, agent_id):
        """
        Calculates reward based on Time Loss (Total Delay) rather than just waiting time.
        This ensures the AI detects and penalizes 'creeping' or slow-moving traffic.
        """
        total_penalty = 0
        controlled_lanes = traci.trafficlight.getControlledLanes(agent_id)
        
        # Check every vehicle near the intersection
        for v_id in traci.vehicle.getIDList():
            if traci.vehicle.getLaneID(v_id) in controlled_lanes:
                v_type = traci.vehicle.getTypeID(v_id)
                
                # REWARD UPGRADE: getTimeLoss captures delay even if speed > 0.1 m/s
                # It effectively measures the 'Road Rage' of drivers moving too slowly.
                delay = traci.vehicle.getTimeLoss(v_id)
                
                # Maintain your research-based weights
                weight = self.vtype_weights.get(v_type, 1.0)
                total_penalty -= (delay * weight)
        
        # Normalized by 1000 because TimeLoss values are cumulative and larger
        return total_penalty / 500.0

    def step(self, actions):
        for agent_id, action in actions.items():
            traci.trafficlight.setPhase(agent_id, action)
            
        # 5-second "Decision Interval"
        for _ in range(10): 
            traci.simulationStep()
            
        states = {a: self._get_state(a) for a in self.agent_ids}
        rewards = {a: self._calculate_reward(a) for a in self.agent_ids}
        
        # Termination logic: end at 3600 steps (1 hour sim)
        done = traci.simulation.getTime() >= 3600 or traci.simulation.getMinExpectedNumber() <= 0
        dones = {a: done for a in self.agent_ids}
        
        return states, rewards, dones, {}

    def reset(self):
        if traci.isLoaded():
            traci.close()
            
        # Silence all SUMO internal logs and warnings for maximum speed
        traci.start([
            self.sumo_binary, "-c", self.sumo_cfg, 
            "--waiting-time-memory", "1000",
            "--lateral-resolution", "0.2",
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--error-log", "sumo_errors.log", # Redirects errors to a file instead of terminal
            "--step-length", "0.5",
        ])
        return {a: self._get_state(a) for a in self.agent_ids}

    def close(self):
        if traci.isLoaded():
            traci.close()