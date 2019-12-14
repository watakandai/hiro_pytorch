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
from .utils import get_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TD3Actor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, scale):
        super(TD3Actor, self).__init__()
        if scale is None:
            scale = torch.ones(state_dim)
        else:
            scale = get_tensor(scale)
        self.scale = nn.Parameter(scale.clone().detach().float(), requires_grad=False)

        self.l1 = nn.Linear(state_dim + goal_dim, 300)
        self.l2 = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, action_dim)

    def forward(self, state, goal):
        a = F.relu(self.l1(torch.cat([state, goal], 1)))
        a = F.relu(self.l2(a))
        return self.scale * torch.tanh(self.l3(a))

class TD3Critic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super(TD3Critic, self).__init__()
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

        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)

        return q

class TD3(object):
    def __init__(
            self,
            state_dim,
            goal_dim,
            action_dim,
            scale,
            model_path,
            actor_lr=0.0001,
            critic_lr=0.001,
            expl_noise=0.1,
            policy_noise=0.2,
            noise_clip=0.5,
            gamma=0.99,
            policy_freq=2,
            tau=0.005):
        self.name = ''
        self.scale = scale
        self.model_path = model_path

        # parameters
        self.expl_noise = expl_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.gamma = gamma
        self.policy_freq = policy_freq
        self.tau = tau

        self.actor = TD3Actor(state_dim, goal_dim, action_dim, scale=scale).to(device)
        self.actor_target = TD3Actor(state_dim, goal_dim, action_dim, scale=scale).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic1 = TD3Critic(state_dim, goal_dim, action_dim).to(device)
        self.critic2 = TD3Critic(state_dim, goal_dim, action_dim).to(device)
        self.critic1_target = TD3Critic(state_dim, goal_dim, action_dim).to(device)
        self.critic2_target = TD3Critic(state_dim, goal_dim, action_dim).to(device)

        self.critic1_optimizer = torch.optim.Adam(
                self.critic1.parameters(),
                lr=critic_lr, weight_decay=0.0001
        )
        self.critic2_optimizer = torch.optim.Adam(
                self.critic2.parameters(),
                lr=critic_lr, weight_decay=0.0001
        )
        self._initialize_target_networks()

        self._initialized = False
        self.total_it = 0

    def _initialize_target_networks(self):
        self._update_target_network(self.critic1_target, self.critic1, 1.0)
        self._update_target_network(self.critic2_target, self.critic2, 1.0)
        self._update_target_network(self.actor_target, self.actor, 1.0)
        self._initialized = True

    def _update_target_network(self, target, origin, tau):
        for target_param, origin_param in zip(target.parameters(), origin.parameters()):
            target_param.data.copy_(tau * origin_param.data + (1.0 - tau) * target_param.data)

    def save(self, timestep=-1):
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.mkdir(os.path.dirname(self.model_path))

        if timestep>0:
            model_path = self.model_path + self.name + '_%i'%(timestep)
        else:
            model_path = self.model_path + self.name

        torch.save(self.actor.state_dict(), model_path+"_actor.h5")
        torch.save(self.actor_optimizer.state_dict(), model_path+"_actor_optimizer.h5")
        torch.save(self.critic1.state_dict(), model_path+"_critic1.h5")
        torch.save(self.critic2.state_dict(), model_path+"_critic2.h5")
        torch.save(self.critic1_optimizer.state_dict(), model_path+"_critic1_optimizer.h5")
        torch.save(self.critic2_optimizer.state_dict(), model_path+"_critic2_optimizer.h5")

    def load(self, timestep=-1):
        if timestep>0:
            model_path = self.model_path + self.name + '_%i'%(timestep)
        else:
            model_path = self.model_path + self.name

        self.actor.load_state_dict(torch.load(model_path+"_actor.h5"))
        self.actor_optimizer.load_state_dict(torch.load(model_path+"_actor_optimizer.h5"))
        self.critic1.load_state_dict(torch.load(model_path+"_critic1.h5"))
        self.critic2.load_state_dict(torch.load(model_path+"_critic2.h5"))
        self.critic1_optimizer.load_state_dict(torch.load(model_path+"_critic1_optimizer.h5"))
        self.critic2_optimizer.load_state_dict(torch.load(model_path+"_critic2_optimizer.h5"))

    def _train(self, states, goals, actions, rewards, n_states, n_goals, not_done):
        self.total_it += 1
        with torch.no_grad():
            noise = (
                torch.randn_like(actions) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            n_actions = self.actor_target(n_states, n_goals) + noise
            n_actions = torch.min(n_actions,  self.actor.scale)
            n_actions = torch.max(n_actions, -self.actor.scale)

            target_Q1 = self.critic1_target(n_states, n_goals, n_actions)
            target_Q2 = self.critic2_target(n_states, n_goals, n_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + not_done * self.gamma * target_Q
            target_Q_detached = target_Q.detach()

        current_Q1 = self.critic1(states, goals, actions)
        current_Q2 = self.critic2(states, goals, actions)

        critic1_loss = F.mse_loss(current_Q1, target_Q_detached)
        critic2_loss = F.mse_loss(current_Q2, target_Q_detached)
        critic_loss = critic1_loss + critic2_loss

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            a = self.actor(states, goals)
            Q1 = self.critic1(states, goals, a)
            actor_loss = -Q1.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._update_target_network(self.critic1_target, self.critic1, self.tau)
            self._update_target_network(self.critic2_target, self.critic2, self.tau)
            self._update_target_network(self.actor_target, self.actor, self.tau)

            return {'actor_loss': actor_loss, 'critic_loss': critic_loss}

        return {'critic_loss': critic_loss}

    def train(self, replay_buffer, iterations=1):
        raise NotImplementedError('Need to implement train function for each controller')

    def policy(self, state, goal, to_numpy=True):
        state = get_tensor(state)
        goal = get_tensor(goal)
        action = self.actor(state, goal)

        if to_numpy:
            return action.cpu().data.numpy().squeeze()

        return action.squeeze()

    def policy_with_noise(self, state, goal, to_numpy=True):
        state = get_tensor(state)
        goal = get_tensor(goal)
        action = self.actor(state, goal)

        action = action + self._sample_exploration_noise(action)
        action = torch.min(action,  self.actor.scale)
        action = torch.max(action, -self.actor.scale)

        if to_numpy:
            return action.cpu().data.numpy().squeeze()

        return action.squeeze()

    def _sample_exploration_noise(self, actions):
        scale = self.expl_noise * self.actor.scale
        mean = torch.zeros(actions.size()).to(device)
        var = torch.ones(actions.size()).to(device)
        return torch.normal(mean, scale*var)

class HigherController(TD3):
    def __init__(
        self,
        state_dim,
        goal_dim,
        action_dim,
        scale,
        model_path,
        actor_lr=0.0001,
        critic_lr=0.001,
        expl_noise=1.0,
        policy_noise=0.2,
        noise_clip=0.5,
        gamma=0.99,
        policy_freq=2,
        tau=0.005):
        super(HigherController, self).__init__(
            state_dim, goal_dim, action_dim, scale, model_path,
            actor_lr, critic_lr, expl_noise, policy_noise,
            noise_clip, gamma, policy_freq, tau
        )
        self.name = '_high'

    def off_policy_corrections(self, low_con, batch_size, sgoals, states, actions, candidate_goals=8):
        first_s = [s[0] for s in states] # First x
        last_s = [s[-1] for s in states] # Last x

        # Shape: (batch_size, 1, subgoal_dim)
        # diff = 1
        diff_goal = (np.array(last_s) -
                     np.array(first_s))[:, np.newaxis, :self.action_dim]

        # Shape: (batch_size, 1, subgoal_dim)
        # original = 1
        # random = candidate_goals
        original_goal = np.array(sgoals)[:, np.newaxis, :]
        random_goals = np.random.normal(loc=diff_goal, scale=.5*self.scale[None, None, :],
                                        size=(batch_size, candidate_goals, original_goal.shape[-1]))
        random_goals = random_goals.clip(-self.scale, self.scale)

        # Shape: (batch_size, 10, subgoal_dim)
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

    def train(self, replay_buffer, low_con):
        if not self._initialized:
            self._initialize_target_networks()

        states, goals, actions, n_states, rewards, not_done, states_arr, actions_arr = replay_buffer.sample()

        actions = self.off_policy_corrections(
            low_con,
            replay_buffer.batch_size,
            actions,
            states_arr,
            actions_arr)

        return self._train(states, goals, actions, rewards, n_states, goals, not_done)

class LowerController(TD3):
    def __init__(
        self,
        state_dim,
        goal_dim,
        action_dim,
        scale,
        model_path,
        actor_lr=0.0001,
        critic_lr=0.001,
        expl_noise=1.0,
        policy_noise=0.2,
        noise_clip=0.5,
        gamma=0.99,
        policy_freq=2,
        tau=0.005):
        super(LowerController, self).__init__(
            state_dim, goal_dim, action_dim, scale, model_path,
            actor_lr, critic_lr, expl_noise, policy_noise,
            noise_clip, gamma, policy_freq, tau
        )
        self.name = '_low'

    def train(self, replay_buffer):
        if not self._initialized:
            self._initialize_target_networks()

        states, sgoals, actions, rewards, n_states, n_sgoals, not_done = replay_buffer.sample()

        return self._train(states, sgoals, actions, rewards, n_states, n_sgoals, not_done)

class HiroAgent():
    def __init__(
        self,
        state_dim,
        action_dim,
        goal_dim,
        subgoal_dim,
        scale_low,
        scale_high,
        model_path,
        buffer_size,
        batch_size,
        low_buffer_freq,
        high_buffer_freq, # c steps, not specified in the paper
        low_train_freq,
        high_train_freq,
        c=10,
        manager_reward_scale=0.1):

        self.high_con = HigherController(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=subgoal_dim,
            scale=scale_high,
            model_path=model_path
            )

        self.low_con = LowerController(
            state_dim=state_dim,
            goal_dim=subgoal_dim,
            action_dim=action_dim,
            scale=scale_low,
            model_path=model_path
            )

        self.replay_buffer_low = LowReplayBuffer(
            state_dim=state_dim,
            goal_dim=subgoal_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            batch_size=batch_size
            )

        self.replay_buffer_high = HighReplayBuffer(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=subgoal_dim,
            buffer_size=buffer_size,
            batch_size=batch_size,
            freq=high_buffer_freq
            )

        self.low_buffer_freq = low_buffer_freq
        self.high_buffer_freq = high_buffer_freq
        self.low_train_freq = low_train_freq
        self.high_train_freq = high_train_freq
        # TODO: Implement reward scaling
        self.low_reward_scale = low_reward_scale
        self.high_reward_scale = high_reward_scale

        self.buf = None
        self.used_high_policy = False

    def append(self, curr_step, s, fg, a, sg, n_s, n_sg, r, d):
        # Low Replay Buffer
        if curr_step % self.low_buffer_freq == 1:
            sr = self.low_reward(s, sg, n_s)
            self.replay_buffer_low.append(
                s, sg, a, sr, n_s, n_sg, d)

        # High Replay Buffer
        if curr_step % self.high_buffer_freq == 1:
            if self.buf:
                self.buf[4] = s
                self.buf[5] = d
                self.replay_buffer_high.append(
                    state=self.buf[0],
                    goal=self.buf[1],
                    action=self.buf[2],
                    n_state=self.buf[4],
                    reward=self.buf[3],
                    done=self.buf[5],
                    state_arr=np.array(self.buf[6]),
                    action_arr=p.array(self.buf[7])
                )
            self.buf = [s, fg, sg, 0, None, None, [], []]

        self.buf[3] += r
        self.buf[6].append(s)
        self.buf[7].append(a)

    def train(self, curr_step):
        losses = {}

        if curr_step % self.low_train_freq == 0:
            loss = self.low_con.train(self.low_replay_buffer)
            losses.update(loss)

        if curr_step % self.high_train_freq == 0:
            loss = self.high_con.train(self.high_replay_buffer, self.low_con)
            losses.update(loss)

        return losses

    def choose_action_with_noise(self, s, sg):
        return self.low_con.policy_with_noise(s, sg)

    def choose_subgoal_with_noise(self, curr_step, s, fg, sg):
        if curr_step % self.high_buffer_freq == 1:
            sg = self.high_con.policy_with_noise(s, fg)
        else:
            sg = self.subgoal_transition(s, sg, n_s)

        return sg

    def choose_action(self, s, sg):
        return self.low_con.policy(s, sg)

    def choose_subgoal(self, curr_step, s, fg, sg):
        if curr_step % self.high_buffer_freq == 1:
            sg = self.high_con.policy(s, fg)
        else:
            sg = self.subgoal_transition(s, sg, n_s)

        return sg

    def subgoal_transition(self, s, sg, n_s):
        return s[:sg.shape[0]] + sg - n_s[:sg.shape[0]]

    def low_reward(self, s, sg, n_s):
        return -np.linalg.norm(s[:sg.shape[0]] + sg - n_s[:sg.shape[0]])

    def save(self, timestep):
        self.low_con.save(timestep)
        self.high_con.save(timestep)

    def load(self, timestep):
        self.low_con.load(timestep)
        self.high_con.load(timestep)

    def evaluate_policy(self, env, eval_episodes=10, render=False, save_video=False, sleep=-1):
        if save_video:
            from OpenGL import GL
            env = gym.wrappers.Monitor(env, directory='video',
                                    write_upon_reset=True, force=True, resume=True, mode='evaluation')
            render = False

        rewards = []
        for e in range(eval_episodes):
            obs = env.reset()
            fg = obs['desired_goal']
            s = obs['observation']
            sg = subgoal.action_space.sample()
            done = False
            steps = 1
            reward_episode_sum = 0

            while not done:
                if render:
                    env.render()
                if sleep>0:
                    time.sleep(sleep)
                sg = self.choose_subgoal(steps, s, fg, sg)
                a = self.choose_action(s, sg)
                n_s, r, done, _ = env.step(a)

                s = n_s
                reward_episode_sum += r
                steps += 1
            else:
                print("Rewards in Episode %i: %.2f"%(e, reward_episode_sum/steps))
                rewards.append(reward_episode_sum)

        return np.array(rewards)
