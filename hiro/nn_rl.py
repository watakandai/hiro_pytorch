import argparse
import os 
import time
import random
from collections import deque
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        pass

    def forward(self, x):
        pass

class Agent():
    def __init__(self, env, epsilon, gamma, lr, model_path):
        self.env = env
        self.actions = list(range(env.action_space.n))
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.model_path = model_path
        self.net = Net()

    def update(self, experiences):
        raise NotImplementedError()

    def policy(self, s, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        
        if np.random.random() < epsilon:
            a = np.random.randint(len(self.actions))
            return torch.tensor([a], device=device), 0 

        return self.predict(s)

    def predict(self, s):
        with torch.no_grad():
            max_q = self.net(s.unsqueeze(0)).max(1)
            return max_q[1], max_q[0].item()
    
    def save(self):
        torch.save(self.net.state_dict(), self.model_path)

    @classmethod
    def load(cls, env, model_path):
        agent = cls(env, epsilon=0, gamma=0, lr=0, model_path=model_path)
        agent.net.load_state_dict(torch.load(model_path))
        agent.net.eval()
        return agent

    def play(self, episodes=5, render=True, sleep=-1):
        for e in range(episodes):
            s = self.env.reset()
            done = False
            rewards = 0
            while not done:
                if render:
                    self.env.render()
                if sleep>0:
                    time.sleep(sleep)
                a, qvalue = self.policy(s, epsilon=0)
                n_s, r, done, info = self.env.step(a.item())
                rewards += r.item()
                s = n_s
            else:
                print("Rewards %.2f"%(rewards))

class Experience():
    def __init__(self, s, a, r, n_s, d):
        self.s = s
        self.a = a
        self.r = r
        self.n_s = n_s
        self.d = d

class ReplayBuffer():
    def __init__(self, buffer_size, batch_size=32):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experiences = deque(maxlen=buffer_size)

    def append(self, experience):
        self.experiences.append(experience)

    def sample(self):
        return random.sample(self.experiences, self.batch_size)

    def is_full(self):
        return len(self.experiences) == self.buffer_size

class Observer():
    def __init__(self, env):
        self._env = env

    @property
    def action_space(self):
        return self._env.action_space
    
    @property
    def observation_space(self):
        return self._env.action_space

    def reset(self):
        return self.transform(self._env.reset())
        
    def render(self, mode="human"):
        self._env.render(mode=mode)

    def step(self, action):
        n_state, reward, done, info = self._env.step(action)
        reward = torch.tensor([reward], device=device, dtype=torch.float)
        return self.transform(n_state), reward, done, info

    def transform(self, state):
        raise NotImplementedError()    

class Epsilon():
    def __init__(self, initial_epsilon, end_epsilon, episodes):
        self.initial_epsilon = initial_epsilon
        self.end_epsilon = end_epsilon
        self.episodes = episodes
    
    def step(self, episode):
        raise NotImplementedError()


class Trainer():
    def __init__(self, 
            env,
            init_eps,
            end_eps,
            buffer_size, 
            batch_size, 
            gamma, 
            lr,
            episodes,
            print_freq, 
            writer_freq, 
            target_network_freq,
            log_path):
        self.env = env
        self.init_eps = init_eps 
        self.end_eps = end_eps 
        self.buffer_size = buffer_size 
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.episodes = episodes
        self.print_freq = print_freq
        self.writer_freq = writer_freq
        self.target_network_freq = target_network_freq
        self.episode = 0
        self.step = 0
        self.loss = 0
        self.reward = 0
        self.qvalue = 0
        self.loss_log = []
        self.reward_log = []
        self.qvalue_log = []
        self.logger = Logger(log_path)

class Logger():
    def __init__(self, log_path):
        self.writer = SummaryWriter(log_path)

    def print(self, name, value, episode=-1, step=-1):
        string = "{} is {}".format(name, value)
        if episode > 0:
            print('Episode:{}, {}'.format(episode, string))
        if step > 0:
            print('Step:{}, {}'.format(step, string))

    def write(self, name, value, index):
        self.writer.add_scalar(name, value, index)