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
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from hiro.nn_rl import Net, Agent, ReplayBuffer, Logger 
from hiro.models import HiroAgent

# TD3
# - Clipped Double Q-Learning
#   - min(Q1, Q2)
# - Delayed Policy Updates 
#   - Update Policy once in 2 Q Updates
# - Target Policy Smoothing
#   - 1st Low pass filter

# Environment and Environment Wrapper
# ReplayBuffer
# low_level_reawrd
# low_level_goal
# relabeling the high-level transition
# add_noise

class HiroTrainer():
    def __init__(self,
            env,
            args,
            buffer_size=200000, 
            batch_size=100, 
            gamma=0.99, 
            episodes=50000,
            print_freq=10, 
            writer_freq=10,
            initial_episodes=5,
            low_buffer_freq=1,
            high_buffer_freq=10,            # TODO: Need to experiment
            low_train_freq=1,
            high_train_freq=10,
            log_path='logs/hiro_pytorch20191202'):
        self.env = env
        self.args = args
        self.buffer_size = buffer_size 
        self.batch_size = batch_size
        self.gamma = gamma
        self.episodes = episodes
        self.print_freq = print_freq
        self.writer_freq = writer_freq
        self.initial_episodes = initial_episodes

        self.episode = 1
        self.step = 1
        self.accum_step = 1
        self.low_critic_loss = 0
        self.low_actor_loss = 0
        self.high_critic_loss = 0
        self.high_actor_loss = 0
        self.high_reward = 0
        self.low_reward = 0
        self.low_critic_log = []
        self.low_actor_log = []
        self.high_critic_log = []
        self.high_actor_log = []
        self.high_reward_log = []
        self.low_reward_log = []
        self.action_logs = []
        self.logger = Logger(log_path)
        self.best_reward = -np.inf
 
        self.agent = HiroAgent(
            env, 
            buffer_size,
            batch_size,
            low_buffer_freq,
            high_buffer_freq,
            low_train_freq,
            high_train_freq)
   

    def train(self, render=False, sleep=-1):
        for e in np.arange(self.episodes)+1:
            self.begin_episode()
            next_tuple = self.env.reset()
            final_goal = next_tuple['desired_goal']
            s = next_tuple['observation']

            low_goal = self.agent.high_con.policy(s, final_goal)
            done = False
            losses = [0, 0, 0, 0]
            
            while not done:
                if render:
                    self.env.render()
                if sleep>0:
                    time.sleep(sleep)

                a = self.agent.low_con.policy(s, low_goal)
                a = self.agent.augment_with_noise(a, self.agent.low_sigma)
                
                # Interact with environment 
                next_tuple, high_r, done, info = self.env.step(a) # add goal

                n_s = next_tuple['observation']

                if self.step % self.agent.c == 0:
                    n_low_goal = self.agent.high_con.policy(n_s, final_goal)
                else:
                    n_low_goal = self.agent.subgoal_transition(s, low_goal, n_s)
                #n_low_goal = self.agent.augment_with_noise(n_low_goal, self.agent.high_sigma)

                low_r = self.agent.low_reward(s, low_goal, n_s)

                self.agent.append(self.step, s, low_goal, final_goal, a, n_s, n_low_goal, low_r, high_r, done, info)

                for i in range(len(low_goal)):
                    self.logger.write('action/high_action%i'%(i), low_goal[i], self.accum_step)

                for i in range(len(a)):
                    self.logger.write('action/action%i'%(i), a[i], self.accum_step)

                self.step += 1
                self.accum_step += 1
                self.high_reward += high_r
                self.low_reward += low_r
                
                s = n_s
                low_goal = n_low_goal
                if e >= self.initial_episodes:
                    self.agent.train(self.step)
                    #losses = self.agent.train(self.step)

 #               self.low_critic_loss += losses[0] 
  #              self.low_actor_loss += losses[1] 
   #             self.high_critic_loss += losses[2] 
    #            self.high_actor_loss += losses[3]
                self.action_logs.append(a)
                
            self.end_episode()

    def begin_episode(self):
        self.low_critic_loss = 0
        self.low_actor_loss = 0
        self.high_critic_loss = 0
        self.high_actor_loss = 0
        self.high_reward = 0
        self.low_reward = 0
        self.step = 1

    def end_episode(self):
        self.low_critic_log.append(self.low_critic_loss/self.step)
        self.low_actor_log.append(self.low_actor_loss/self.step)
        self.high_critic_log.append(self.high_critic_loss/self.step)
        self.high_actor_log.append(self.high_actor_loss/self.step)
        self.high_reward_log.append(self.high_reward/self.step)
        self.low_reward_log.append(self.low_reward/self.step)

        if _is_update(self.episode, self.print_freq):
            # get start index
            s = self.episode - self.print_freq
            # get mean of start:end logs
            self.high_reward = np.mean(self.high_reward_log[s:])
            # print to console
            self.logger.print('reward', self.high_reward, self.episode)
        
        if _is_update(self.episode, self.writer_freq):
            # get start index
            s = self.episode - self.writer_freq
            # get mean of start:end logs
            low_critic_loss = np.mean(self.low_critic_log[s:])
            low_actor_loss = np.mean(self.low_actor_log[s:])
            high_critic_loss = np.mean(self.high_critic_log[s:])
            high_actor_loss = np.mean(self.high_actor_log[s:])
            high_reward = np.mean(self.high_reward_log[s:])
            low_reward = np.mean(self.low_reward_log[s:])
            # log to tensorboard 
            self.logger.write('loss/low critic loss', low_critic_loss, self.episode)
            self.logger.write('loss/low actor loss', low_actor_loss, self.episode)
            self.logger.write('loss/high critic loss', high_critic_loss, self.episode)
            self.logger.write('loss/high actor loss', high_actor_loss, self.episode)
            self.logger.write('reward/reward', high_reward, self.episode)
            self.logger.write('reward/low_reward', low_reward, self.episode)
       
        # save model if it performed well
        if self.high_reward > self.best_reward:
            self.agent.save()
            self.best_reward = self.high_reward
        
        # increment episode 
        self.episode += 1

def _is_update(episode, freq):
    if episode!=0 and episode%freq==0:
        return True
    return False

