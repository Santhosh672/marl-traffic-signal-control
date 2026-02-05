import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        m.bias.data.fill_(0.0)

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.apply(init_weights)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.action_head(x), dim=-1)

class CriticNetwork(nn.Module):
    """
    Enhanced Critic for 9-junction coordination.
    Uses 3 layers to process the 108-dimensional global state.
    """
    def __init__(self, global_obs_dim, hidden_dim=256): 
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(global_obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.value_head = nn.Linear(hidden_dim // 2, 1)
        self.apply(init_weights)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.value_head(x)

class MAPPOAgent:
    def __init__(self, agent_id, obs_dim, action_dim, global_obs_dim):
        self.agent_id = agent_id
        self.actor = ActorNetwork(obs_dim, action_dim).to(device)
        self.critic = CriticNetwork(global_obs_dim).to(device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    def select_action(self, obs):
        state = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def save_model(self, folder="output/model"): # Updated default
        import os
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{folder}/actor_{self.agent_id}.pth")
        torch.save(self.critic.state_dict(), f"{folder}/critic_{self.agent_id}.pth") # Added Critic