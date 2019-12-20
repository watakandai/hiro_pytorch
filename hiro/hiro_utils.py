import torch
import numpy as np
from hiro.models import HiroAgent, TD3Agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def spawn_dims( env):
    goal_dim = 2
    state_dim = env.state_dim
    action_dim = env.action_dim
    scale = env.action_space.high * np.ones(action_dim)

    return state_dim, goal_dim, action_dim, scale

def spawn_hiro(args, env, subgoal):
    state_dim, goal_dim, action_dim, scale_low = spawn_dims(env)

    subgoal_dim = subgoal.action_dim
    scale_high = subgoal.action_space.high * np.ones(subgoal_dim)

    agent = HiroAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        subgoal_dim=subgoal_dim,
        scale_low=scale_low,
        scale_high=scale_high,
        model_path='_'.join([args.model_path, args.filename]),
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        buffer_freq=args.hbuffer_freq,
        train_freq=args.train_freq,
        reward_scaling=args.reward_scaling,
        policy_freq_high=args.policy_freq_high,
        policy_freq_low=args.policy_freq_low
        )
    return agent

def spawn_td3(args, env):
    state_dim, goal_dim, action_dim, scale = spawn_dims(env)
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        scale=scale,
        model_path='_'.join([args.model_path, args.filename]),
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        )
    return agent

class ReplayBuffer():
    def __init__(self, state_dim, goal_dim, action_dim, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((buffer_size, state_dim))
        self.goal = np.zeros((buffer_size, goal_dim))
        self.action = np.zeros((buffer_size, action_dim))
        self.n_state = np.zeros((buffer_size, state_dim))
        self.reward = np.zeros((buffer_size, 1))
        self.not_done = np.zeros((buffer_size, 1))

        self.device = device

    def append(self, state, goal, action, n_state, reward, done):
        self.state[self.ptr] = state
        self.goal[self.ptr] = goal
        self.action[self.ptr] = action
        self.n_state[self.ptr] = n_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr+1) % self.buffer_size
        self.size = min(self.size+1, self.buffer_size)

    def sample(self):
        ind = np.random.randint(0, self.size, size=self.batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.goal[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.n_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )

class LowReplayBuffer(ReplayBuffer):
    def __init__(self, state_dim, goal_dim, action_dim, buffer_size, batch_size):
        super(LowReplayBuffer, self).__init__(state_dim, goal_dim, action_dim, buffer_size, batch_size)
        self.n_goal = np.zeros((buffer_size, goal_dim))

    def append(self, state, goal, action, n_state, n_goal, reward, done):
        self.state[self.ptr] = state
        self.goal[self.ptr] = goal
        self.action[self.ptr] = action
        self.n_state[self.ptr] = n_state
        self.n_goal[self.ptr] = n_goal
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr+1) % self.buffer_size
        self.size = min(self.size+1, self.buffer_size)

    def sample(self):
        ind = np.random.randint(0, self.size, size=self.batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.goal[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.n_state[ind]).to(self.device),
            torch.FloatTensor(self.n_goal[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )

class HighReplayBuffer(ReplayBuffer):
    def __init__(self, state_dim, goal_dim, subgoal_dim, action_dim, buffer_size, batch_size, freq):
        super(HighReplayBuffer, self).__init__(state_dim, goal_dim, action_dim, buffer_size, batch_size)
        self.action = np.zeros((buffer_size, subgoal_dim))
        self.state_arr = np.zeros((buffer_size, freq, state_dim))
        self.action_arr = np.zeros((buffer_size, freq, action_dim))

    def append(self, state, goal, action, n_state, reward, done, state_arr, action_arr):
        self.state[self.ptr] = state
        self.goal[self.ptr] = goal
        self.action[self.ptr] = action
        self.n_state[self.ptr] = n_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.state_arr[self.ptr,:,:] = state_arr
        self.action_arr[self.ptr,:,:] = action_arr

        self.ptr = (self.ptr+1) % self.buffer_size
        self.size = min(self.size+1, self.buffer_size)

    def sample(self):
        ind = np.random.randint(0, self.size, size=self.batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.goal[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.n_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.state_arr[ind]).to(self.device),
            torch.FloatTensor(self.action_arr[ind]).to(self.device)
        )

class SubgoalActionSpace(object):
    def __init__(self):
        self.shape = (15,1)
        self.low = np.array((-10, -10, -0.5, -1, -1, -1, -1,
                -0.5, -0.3, -0.5, -0.3, -0.5, -0.3, -0.5, -0.3))
        self.high = -self.low

    def sample(self):
        return (self.high - self.low) * np.random.sample() + self.low

class Subgoal(object):
    def __init__(self):
        self.action_space = SubgoalActionSpace()
        self.action_dim = self.action_space.shape[0]
