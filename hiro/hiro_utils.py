import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    def __init__(self, dim):
        limits = np.array([-10, -10, -0.5, -1, -1, -1, -1,
                    -0.5, -0.3, -0.5, -0.3, -0.5, -0.3, -0.5, -0.3])
        self.shape = (dim,1)
        self.low = limits[:dim]
        self.high = -self.low

    def sample(self):
        return (self.high - self.low) * np.random.sample() + self.low

class Subgoal(object):
    def __init__(self, dim=15):
        self.action_space = SubgoalActionSpace(dim)
        self.action_dim = self.action_space.shape[0]
