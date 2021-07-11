import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256,
            name='critic'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name

        self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims)
        self.b1=nn.BatchNorm1d(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.b2=nn.BatchNorm1d(self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        state=T.tensor(state,dtype=T.float).to(self.device)
        action=T.tensor(action,dtype=T.float).to(self.device)
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value=self.b1(action_value)
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value=self.b2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q



class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims,  fc1_dims=256,
            fc2_dims=256, n_actions=1):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions



        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.b1=nn.BatchNorm1d(self.fc1_dims)
        self.b2=nn.BatchNorm1d(self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        state=T.tensor(state,dtype=T.float).to(self.device)
        prob = self.fc1(state)
        prob=self.b1(prob)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob=self.b2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=1e-6, max=1)

        return mu, sigma

    def sample_normal(self, state,train=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        actions = probabilities.rsample()
        actions=T.sigmoid(actions)*2
        log_probs = probabilities.log_prob(actions)*(actions)*(1-actions)
        return actions, log_probs



