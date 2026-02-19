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
        
        self.agent_ids = []
        self.last_action = {} 
        
        # Heterogeneous weights for Indian traffic mix
        self.vtype_weights = {
            "car": 1.0, "motorcycle": 0.5, "auto_rickshaw": 1.5,
            "truck": 3.0, "bus": 5.0, "lcv": 2.0
        }

        self.observation_spaces = {}
        self.action_spaces = {}

    def _get_state(self, agent_id):
        """Extracts 12-feature state vector. Sensor range increased to 250m."""
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
                        # Increased to 250m to capture gridlock earlier
                        if (edge_len - lane_pos) < 250: 
                            vehs_on_approach.append(v)

            count = len(vehs_on_approach)
            avg_speed = np.mean([traci.vehicle.getSpeed(v) for v in vehs_on_approach]) if vehs_on_approach else 0
            state_vector.extend([min(count / 100.0, 1.0), min(avg_speed / 25.0, 1.0)])
            
        return np.array(state_vector, dtype=np.float32)

    def _calculate_reward(self, agent_id, current_action):
        """
        Hybridized Reward: Pressure + Max Wait Time (Starvation Prevention) + 
        Reduced Stability Penalty.
        """
        total_penalty = 0
        
        # 1. Reduced Stability Penalty (Softened from -50 to -5)
        # Encourages exploration of new phases to find 'Green' solutions
        if current_action != self.last_action.get(agent_id, 0):
            total_penalty -= 5.0 
            
        controlled_lanes = traci.trafficlight.getControlledLanes(agent_id)
        
        # 2. Max Waiting Time (Logic from Graph Modeling article)
        # Prevents starvation by penalizing based on the vehicle waiting the longest
        lane_leader_waits = []
        for lane in controlled_lanes:
            vehs = traci.lane.getLastStepVehicleIDs(lane)
            if vehs:
                # The last vehicle in the ID list is the leader closest to the junction
                leader_veh = vehs[-1] 
                lane_leader_waits.append(traci.vehicle.getWaitingTime(leader_veh))
        
        if lane_leader_waits:
            max_wait = min(max(lane_leader_waits), 120) # Capped at 120s
            total_penalty -= (max_wait * 2.0) # Weighted to force clearing the queue

        # 3. Existing Pressure & Heterogeneous Weights
        for v_id in traci.vehicle.getIDList():
            if traci.vehicle.getLaneID(v_id) in controlled_lanes:
                v_type = traci.vehicle.getTypeID(v_id)
                delay = traci.vehicle.getTimeLoss(v_id) 
                wait_time = min(traci.vehicle.getWaitingTime(v_id), 120) 
                pressure = (wait_time ** 2) 
                
                weight = self.vtype_weights.get(v_type, 1.0)
                total_penalty -= ((delay + pressure) * weight)
        
        self.last_action[agent_id] = current_action
        
        # 4. Normalization: Scaled to 100,000 for CTDE Critic stability
        return total_penalty / 100000.0

    def reset(self, label="sim0", port=8813):
        if traci.isLoaded():
            traci.close()
        
        # Redirect error logs to output/log folder
        log_path = "output/log/sumo_errors.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
            
        traci.start([
            self.sumo_binary, "-c", self.sumo_cfg, 
            "--error-log", log_path,
            "--waiting-time-memory", "1000",
            "--lateral-resolution", "0.2",
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--step-length", "0.5",
        ], label=label, port=port)

        conn = traci.getConnection(label)
        
        # --- NEW: Phase Filtering Logic ---
        self.green_phases = {} 
        for agent in self.agent_ids:
            # Retrieve the traffic light logic from the XML
            logic = conn.trafficlight.getAllProgramLogics(agent)[0]
            phases = logic.phases
            
            # Filter for phases that contain 'G' (Green) and lack 'y' (Yellow)
            # This ensures the AI only picks phases that move traffic
            valid_indices = [i for i, p in enumerate(phases) if ('G' in p.state or 'g' in p.state) and 'y' not in p.state]
            
            # Fallback in case a junction has no 'pure green' phases
            if not valid_indices:
                valid_indices = [i for i, p in enumerate(phases) if 'G' in p.state or 'g' in p.state]
                
            self.green_phases[agent] = valid_indices
            
            # The AI action space is now limited to the number of valid Green phases
            self.action_spaces[agent] = Discrete(len(self.green_phases[agent]))
            self.last_action[agent] = 0

        return {a: self._get_state(a) for a in self.agent_ids}

    def step(self, actions):
        """Executes one step, mapping AI actions to XML Green indices."""
        for agent_id, ai_action in actions.items():
            # Map the AI's 0, 1, 2... to actual XML indices (e.g., 0, 4, 8...)
            target_phase = self.green_phases[agent_id][ai_action]
            traci.trafficlight.setPhase(agent_id, target_phase)
            
        # Increased simulation interval to 10 seconds (20 steps * 0.5s)
        # This gives traffic more time to react to signal changes
        for _ in range(20): 
            traci.simulationStep()
            
        states = {a: self._get_state(a) for a in self.agent_ids}
        rewards = {a: self._calculate_reward(a, actions[a]) for a in self.agent_ids}
        
        # Termination at 1800s (30 minutes)
        done = traci.simulation.getTime() >= 1800 or traci.simulation.getMinExpectedNumber() <= 0
        dones = {a: done for a in self.agent_ids}
        
        return states, rewards, dones, {}

    def close(self):
        if traci.isLoaded():
            traci.close()