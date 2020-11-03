import torch
import torch.nn as nn
import torch.nn.functional as F
     
class Actor(nn.Module):
    def __init__(self, state_dim, num_actions, seed, units_fc1=256):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.linear1 = nn.Linear(state_dim, units_fc1) 
        self.linear3 = nn.Linear(units_fc1, num_actions)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = torch.tanh(self.linear3(x))
        return x
    
class Critic(nn.Module):
    def __init__(self, state_dim, num_actions, seed, units_fc1=256, units_fc2=256, units_fc3=128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.linear1 = nn.Linear(state_dim + num_actions, units_fc1)
        self.linear2 = nn.Linear(units_fc1, units_fc2)
        self.linear3 = nn.Linear(units_fc2, units_fc3)
        self.linear4 = nn.Linear(units_fc3, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x