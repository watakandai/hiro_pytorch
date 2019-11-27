##################################################
# @copyright Kandai Watanabe
# @email kandai.wata@gmail.com
# @institute University of Colorado Boulder
# 
# NN Models for HIRO 
# (Data-Efficient Hierarchical Reinforcement Learning)
# Parameters can be find in the original paper
import os 
import copy
import time
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn_rl import Agent, ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def var(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor

def get_tensor(z):
    if len(z.shape) == 1:
        return var(torch.FloatTensor(z.copy())).unsqueeze(0)
    else:
        return var(torch.FloatTensor(z.copy()))

class HigherActor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, max_action, scale=None):
        super(HigherActor, self).__init__()
        if scale is None:
            scale = torch.ones(state_dim + goal_dim)
        self.scale = nn.Parameter(torch.tensor(scale).float(), requires_grad=False)
        self.l1 = nn.Linear(state_dim + goal_dim, 300)
        self.l2 = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state, goal):
        a = F.relu(self.l1(torch.cat([state, goal], 1)))
        a = F.relu(self.l2(a))
        return self.scale * self.max_action * torch.tanh(self.l3(a))

class HigherCritic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super(HigherCritic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + goal_dim + action_dim, 300)
        self.l2 = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + goal_dim + action_dim, 300)
        self.l5 = nn.Linear(300, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, goal, action):
        sa = torch.cat([state, goal, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l1(sa))
        q2 = F.relu(self.l2(q2))
        q2 = self.l3(q2)
        
        return q1, q2

    def Q1(self, state, goal, action):
        sa = torch.cat([state, goal, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1

class LowerActor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, max_action=1):
        super(LowerActor, self).__init__()
        self.l1 = nn.Linear(state_dim + goal_dim, 300)
        self.l2 = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state, goal):
        a = F.relu(self.l1(torch.cat([state, goal], 1)))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class LowerCritic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super(LowerCritic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + goal_dim + action_dim, 300)
        self.l2 = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + goal_dim + action_dim, 300)
        self.l5 = nn.Linear(300, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, goal, action):
        sa = torch.cat([state, goal, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l1(sa))
        q2 = F.relu(self.l2(q2))
        q2 = self.l3(q2)
        
        return q1, q2

    def Q1(self, state, goal, action):
        sa = torch.cat([state, goal, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1

class HigherController():
    def __init__(
        self,
        state_dim,
        goal_dim,
        action_dim,
        max_action,
        model_path,
        scale,
        actor_lr=0.0001,
        critic_lr=0.001,
        policy_noise=0.2,
        noise_clip=0.5,
        gamma=0.99,
        policy_freq=2,
        tau=0.005):

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        #self.max_action = max_action
        self.max_action = max_action = torch.Tensor(max_action).float().to(device)
        self.model_path = model_path
        self.scale = scale
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.gamma = gamma
        self.policy_freq = policy_freq
        self.tau = tau

        self.actor = HigherActor(state_dim, goal_dim, action_dim, max_action, scale).to(device)
        self.actor_target = HigherActor(state_dim, goal_dim, action_dim, max_action, scale).to(device)
        self.actor_target.eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic = HigherCritic(state_dim, goal_dim, action_dim).to(device)
        self.critic_target = HigherCritic(state_dim, goal_dim, action_dim).to(device)
        self.critic_target.eval()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.actor_loss = torch.tensor(0).float().to(device)
        self.critic_loss = torch.tensor(0).float().to(device)

        self._initialize_target_networks()
        self.loss_fn = F.mse_loss 

        self.initalized = False
        self.total_it = 0

    def _initialize_target_networks(self):
        self._update_target_network(self.critic_target, self.critic, 1.0)
        self._update_target_network(self.actor_target, self.actor, 1.0)
        self.initialized = True

    def _update_target_network(self, target, origin, tau):
        for target_param, origin_param in zip(target.parameters(), origin.parameters()):
            target_param.data = tau * origin_param.data + \
                (1.0 - tau) * target_param.data    

    def save(self):
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.mkdir(os.path.dirname(self.model_path))
        torch.save(self.actor.state_dict(), self.model_path+"_high_actor")
        torch.save(self.actor_optimizer.state_dict(), self.model_path+"_high_actor_optimizer")
        torch.save(self.critic.state_dict(), self.model_path+"_high_critic")
        torch.save(self.critic_optimizer.state_dict(), self.model_path+"_high_critic_optimizer")

    def load(self):
        self.actor.load_state_dict(torch.load(self.model_path+"_high_actor"))
        self.actor_optimizer.load_state_dict(torch.load(self.model_path+"_high_actor_optimizer"))
        self.critic.load_state_dict(torch.load(self.model_path+"_high_critic"))
        self.critic_optimizer.load_state_dict(torch.load(self.model_path+"_high_critic_optimizer"))

    def off_policy_corrections(self, low_con, batch_size, low_goals, states, actions, candidate_goals=8):
        first_s = [s[0] for s in states] # First x
        last_s = [s[-1] for s in states] # Last x

        # Shape: (batchsz, 1, subgoaldim)
        diff_goal = (np.array(last_s) -
                     np.array(first_s))[:, np.newaxis, :self.action_dim]

        # Shape: (batchsz, 1, subgoaldim) #TODO: SCALE!!!!!!!!!!!!!
        original_goal = np.array(low_goals)[:, np.newaxis, :]
        random_goals = np.random.normal(loc=diff_goal, scale=.5*self.scale[None, None, :],
                                        size=(batch_size, candidate_goals, original_goal.shape[-1]))
        random_goals = random_goals.clip(-self.scale, self.scale)

        # Shape: (batchsz, 10, subgoal_dim)
        candidates = np.concatenate([original_goal, diff_goal, random_goals], axis=1)
        states = np.array(states)[:, :-1, :]
        actions = np.array(actions)
        seq_len = len(states[0])

        # For ease
        new_batch_sz = seq_len * batch_size
        action_dim = actions[0][0].shape
        obs_dim = states[0][0].shape
        ncands = candidates.shape[1]

        true_actions = actions.reshape((new_batch_sz,) + action_dim)
        observations = states.reshape((new_batch_sz,) + obs_dim)
        goal_shape = (new_batch_sz, self.action_dim)
        # observations = get_obs_tensor(observations, sg_corrections=True)

        # batched_candidates = np.tile(candidates, [seq_len, 1, 1])
        # batched_candidates = batched_candidates.transpose(1, 0, 2)

        policy_actions = np.zeros((ncands, new_batch_sz) + action_dim)

        # TODO: MULTI_SUBOAL_TRANSITION!!!!!!!!!!!
        for c in range(ncands):
            subgoal = candidates[:,c]
            candidate = (subgoal + states[:, 0, :self.action_dim])[:, None] - states[:, :, :self.action_dim]
            candidate = candidate.reshape(*goal_shape)
            policy_actions[c] = low_con.policy(observations, candidate)

        difference = (policy_actions - true_actions)
        difference = np.where(difference != -np.inf, difference, 0)
        difference = difference.reshape((ncands, batch_size, seq_len) + action_dim).transpose(1, 0, 2, 3)

        logprob = -0.5*np.sum(np.linalg.norm(difference, axis=-1)**2, axis=-1)
        max_indices = np.argmax(logprob, axis=-1)

        return candidates[np.arange(batch_size), max_indices]

    def update(self, experiences, low_con):
        self.total_it += 1

        # state, action, reward, next_state, done, next_states_betw, actions_betw
        states = np.array([e[0] for e in experiences if e is not None])
        goals = np.array([e[1] for e in experiences if e is not None])
        low_goals = np.array([e[2] for e in experiences if e is not None])
        rewards = np.array([e[3] for e in experiences if e is not None])
        n_states = np.array([e[4] for e in experiences if e is not None])
        not_done = np.array([1-e[5] for e in experiences if e is not None])
        states_accum = np.array([e[6] for e in experiences if e is not None])
        actions_accum = np.array([e[7] for e in experiences if e is not None])

        low_goals = self.off_policy_corrections(low_con, len(experiences), low_goals, states_accum, actions_accum)

        states = torch.from_numpy(states).float().to(device)
        goals = torch.from_numpy(goals).float().to(device)
        low_goals = torch.from_numpy(low_goals).float().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        n_states = torch.from_numpy(n_states).float().to(device)
        not_done = torch.from_numpy(not_done).float().to(device)
        states_accum = torch.from_numpy(states_accum).float().to(device)
        actions_accum = torch.from_numpy(actions_accum).float().to(device)

        with torch.no_grad():
            noise = (torch.randn_like(low_goals) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
	
            n_actions = torch.max(torch.min(self.actor_target(n_states, goals) + noise, self.max_action), -self.max_action)

            target_Q1, target_Q2 = self.critic_target(n_states, goals, n_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + not_done * self.gamma * target_Q
            target_Q_detouched = target_Q.detouch()

        current_Q1, current_Q2 = self.critic(states, goals, low_goals)

        self.critic_loss = self.loss_fn(current_Q1, target_Q_detouched) + self.loss_fn(current_Q2, target_Q_detouched)

        self.critic_optimizer.zero_grad()
        self.critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            self.actor_loss = -self.critic.Q1(states, goals, self.actor(states, goals)).mean()

            self.actor_optimizer.zero_grad()
            self.actor_loss.backward()
            self.actor_optimizer.step()

            self._update_target_network(self.critic_target, self.critic, self.tau)
            self._update_target_network(self.actor_target, self.actor, self.tau)
        
    def policy(self, state, goal, to_numpy=True):
        state = get_tensor(state)
        goal = get_tensor(goal)

        with torch.no_grad():
            if to_numpy:
                return self.actor(state, goal).cpu().data.numpy().squeeze()
            else:
                return self.actor(state, goal).squeeze()
                
class LowerController():
    def __init__(
        self, 
        state_dim,
        goal_dim,
        action_dim,
        max_action,
        model_path,
        actor_lr=0.0001,
        critic_lr=0.001,
        policy_noise=0.2,
        noise_clip=0.5,
        gamma=0.99,
        policy_freq=2,
        tau=0.005):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.max_action = max_action
        self.model_path = model_path
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.gamma = gamma
        self.policy_freq = policy_freq
        self.tau = tau

        self.actor = LowerActor(state_dim, goal_dim, action_dim, max_action).to(device)
        self.actor_target = LowerActor(state_dim, goal_dim, action_dim, max_action).to(device)
        self.actor_target.eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic = LowerCritic(state_dim, goal_dim, action_dim).to(device)
        self.critic_target = LowerCritic(state_dim, goal_dim, action_dim).to(device)
        self.critic_target.eval()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.actor_loss = torch.tensor(0).float().to(device)
        self.critic_loss = torch.tensor(0).float().to(device)

        self._initialize_target_networks()

        self.loss_fn = F.mse_loss 

        self.initalized = False
        self.total_it = 0

    def _initialize_target_networks(self):
        self._update_target_network(self.critic_target, self.critic, 1.0)
        self._update_target_network(self.actor_target, self.actor, 1.0)
        self.initialized = True

    def _update_target_network(self, target, origin, tau):
        for target_param, origin_param in zip(target.parameters(), origin.parameters()):
            target_param.data = tau * origin_param.data + \
                (1.0 - tau) * target_param.data    

    def save(self):
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.mkdir(os.path.dirname(self.model_path))
        torch.save(self.actor.state_dict(), self.model_path+"_low_actor")
        torch.save(self.actor_optimizer.state_dict(), self.model_path+"_low_actor_optimizer")
        torch.save(self.critic.state_dict(), self.model_path+"_low_critic")
        torch.save(self.critic_optimizer.state_dict(), self.model_path+"_low_critic_optimizer")

    def load(self):
        self.critic.load_state_dict(torch.load(self.model_path+"_low_critic"))
        self.critic_optimizer.load_state_dict(torch.load(self.model_path+"_low_critic_optimizer"))
        self.actor.load_state_dict(torch.load(self.model_path+"_low_actor"))
        self.actor_optimizer.load_state_dict(torch.load(self.model_path+"_low_actor_optimizer"))
        
    def update(self, experiences):
        self.total_it += 1

        # (state, lgoal), a, low_r, (n_s, n_lgoal), float(done)
        """
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(device)
        low_goals = torch.from_numpy(np.vstack([e[1] for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(device)
        n_states = torch.from_numpy(np.vstack([e[4] for e in experiences])).float().to(device)
        n_low_goals = torch.from_numpy(np.vstack([e[5] for e in experiences])).float().to(device)
        not_done = torch.from_numpy(np.vstack([1-e[6] for e in experiences])).float().to(device)
        """
        
        states = torch.tensor([e[0] for e in experiences if e is not None], dtype=torch.float32, device=device)
        low_goals = torch.tensor([e[1] for e in experiences if e is not None], dtype=torch.float32, device=device)
        actions = torch.tensor([e[2] for e in experiences if e is not None], dtype=torch.float32, device=device)
        rewards = torch.tensor([e[3] for e in experiences if e is not None], dtype=torch.float32, device=device)
        n_states = torch.tensor([e[4] for e in experiences if e is not None], dtype=torch.float32, device=device)
        n_low_goals = torch.tensor([e[5] for e in experiences if e is not None], dtype=torch.float32, device=device)
        not_done = torch.tensor([e[6] for e in experiences if e is not None], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            
            n_actions = (self.actor_target(n_states, n_low_goals) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(n_states, n_low_goals, n_actions) 
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + not_done * self.gamma * target_Q           # Need detouch?
            target_Q_detouched = target_Q.detouch()

        current_Q1, current_Q2 = self.critic(states, low_goals, actions)

        self.critic_loss = self.loss_fn(current_Q1, target_Q_detouched) +\
                            self.loss_fn(current_Q2, target_Q_detouched)

        self.critic_optimizer.zero_grad()
        self.critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            self.actor_loss = -self.critic.Q1(states, low_goals, self.actor(states, low_goals)).mean()

            self.actor_optimizer.zero_grad()
            self.actor_loss.backward()
            self.actor_optimizer.step()

            # TODO: Might need to take mean over each loss
            self._update_target_network(self.actor_target, self.actor, self.tau)

        self._update_target_network(self.critic_target, self.critic, self.tau)

    def policy(self, state, goal, to_numpy=True):
        state = get_tensor(state)
        goal = get_tensor(goal)

        with torch.no_grad():
            if to_numpy:
                return self.actor(state, goal).cpu().data.numpy().squeeze()
            else:
                return self.actor(state, goal).squeeze()

# relabeling the high-level transition
# Update
# to(device)
# no_grad()
class HiroAgent():
    def __init__(
        self,
        env,
        buffer_size=200000,
        batch_size=100,
        low_buffer_freq=1,
        high_buffer_freq=10, # c steps, not specified in the paper
        low_train_freq=1,
        high_train_freq=10,
        low_sigma=1,
        high_sigma=1,
        c=10,
        model_path='model/hiro_pytorch.h5'):

        self.env = env

        obs = env.reset()
        goal = obs['desired_goal']
        state = obs['observation']
        state_dim = state.shape[0]
        self.goal_dim = goal.shape[0]
        action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])

        
        low = np.array((-10, -10, -0.5, -1, -1, -1, -1,
                -0.5, -0.3, -0.5, -0.3, -0.5, -0.3, -0.5, -0.3))
        high = -low
        man_scale = (high - low)/2
        high_max_action = high
        self.low_goal_dim = man_scale.shape[0]
        goal_dim = 2

        self.high_con = HigherController(state_dim, goal_dim, self.low_goal_dim, high_max_action, model_path, scale=man_scale)
        self.low_con = LowerController(state_dim, self.low_goal_dim, action_dim, self.max_action, model_path)
        self.high_replay_buffer = ReplayBuffer(buffer_size, batch_size) 
        self.low_replay_buffer = ReplayBuffer(buffer_size, batch_size)  
        self.low_buffer_freq = low_buffer_freq
        self.high_buffer_freq = high_buffer_freq
        self.low_train_freq = low_train_freq
        self.high_train_freq = high_train_freq
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma
        self.c = c

        self.reward_sum = 0

    def append(self, curr_step, s, low_goal, final_goal, a, n_s, n_low_goal, low_r, high_r, done, info):
        self.reward_sum += high_r

        if curr_step % self.low_buffer_freq == 0:
            # (state, lgoal), a, low_r, (n_s, n_lgoal), float(done)
            self.low_replay_buffer.append([
                s, low_goal, 
                a, 
                low_r, 
                n_s, n_low_goal, 
                float(done)
                ])

        if curr_step == 1:
            # state, action, reward, ext_state, done, next_states_betw, actions_betw
            self.high_transition = [s, final_goal, low_goal, 0, None, None, [s], []]

        self.high_transition[3] += high_r
        self.high_transition[6].append(n_s)
        self.high_transition[7].append(a)

        if curr_step % self.high_buffer_freq == 0:
            self.high_transition[4] = s
            self.high_transition[5] = float(done)
            self.high_replay_buffer.append(copy.copy(self.high_transition))
            self.high_transition = [s, final_goal, low_goal, 0, None, None, [s], []]

    def train(self, curr_step):
        if curr_step % self.low_train_freq == 0:
            batch = self.low_replay_buffer.sample()
            self.low_con.update(batch)
        
        if curr_step % self.high_train_freq == 0:
            batch = self.high_replay_buffer.sample()
            self.high_con.update(batch, self.low_con)

        return  self.low_con.critic_loss.cpu().data.numpy(),    \
                self.low_con.actor_loss.cpu().data.numpy(),     \
                self.high_con.critic_loss.cpu().data.numpy(),   \
                self.high_con.actor_loss.cpu().data.numpy()

    def subgoal_transition(self, s, low_g, n_s):
        return s[:self.low_goal_dim] + low_g - n_s[:self.low_goal_dim]

    def low_reward(self, s, low_g, n_s, scale=1):
        return -np.linalg.norm(s[:self.low_goal_dim] + low_g - n_s[:self.low_goal_dim], 1)*scale

    def augment_with_noise(self, action, sigma):
        aug_action = action + np.random.normal(0, sigma, size=action.shape[0])
        return aug_action.clip(-self.max_action, self.max_action)

    def save(self):
        self.low_con.save()
        self.high_con.save()

    # def load(cls, env, model_path):
    def load(self):
        self.low_con.load()
        self.high_con.load()

    def play(self, episodes=5, render=True, sleep=-1):
        for e in range(episodes):
            obs = self.env.reset()

            final_goal = obs['desired_goal']
            now = obs['achieved_goal']
            s = obs['observation']
            print(final_goal)
            print(now)
            print(s)

            low_goal = self.high_con.policy(s, final_goal)
            done = False
            rewards = 0
            steps = 1

            while not done:
                if render:
                    self.env.render()
                if sleep>0:
                    time.sleep(sleep)
                a = self.low_con.policy(s, low_goal)

                obs, r, done, info = self.env.step(a)
                n_s = obs['observation']

                if steps % self.c == 0:
                    n_low_goal = self.high_con.policy(n_s, final_goal)
                else:
                    n_low_goal = self.subgoal_transition(s, low_goal, n_s)

                rewards += r
                s = n_s
                low_goal = n_low_goal
                steps += 1
            else:
                print("Rewards %.2f"%(rewards/steps))