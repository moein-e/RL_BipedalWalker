import torch
import torch.nn as nn
import torch.nn.functional as F
     
class Actor(nn.Module):
    def __init__(self, state_dim, num_actions, seed, neurons1=256):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.linear1 = nn.Linear(state_dim, neurons1) 
        # self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(neurons1, num_actions)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        # x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        # x = self.linear3(x)
        return x
    
class Critic(nn.Module):
    def __init__(self, state_dim, num_actions, seed, neurons1=256, neurons2=256, neurons3=128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.linear1 = nn.Linear(state_dim + num_actions, neurons1)
        self.linear2 = nn.Linear(neurons1, neurons2)
        self.linear3 = nn.Linear(neurons2, neurons3)
        self.linear4 = nn.Linear(neurons3, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x