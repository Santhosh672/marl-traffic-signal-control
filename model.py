import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.action_head(x), dim=-1)

class CriticNetwork(nn.Module):
    def __init__(self, global_obs_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        # total_obs_dim is now the sum of all agents' observations
        self.fc1 = nn.Linear(global_obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.value_head(x)

class MAPPOAgent:
    def __init__(self, agent_id, obs_dim, action_dim, global_obs_dim):
        self.agent_id = agent_id
        self.actor = ActorNetwork(obs_dim, action_dim)
        self.critic = CriticNetwork(global_obs_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    def select_action(self, obs):
        state = torch.FloatTensor(obs).unsqueeze(0)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)