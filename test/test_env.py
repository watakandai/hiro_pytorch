import unittest
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from envs.create_maze_env import create_maze_env

ENV_NAME = 'AntMaze'

class EnvTest(unittest.TestCase):
    def test_dimension(self):
        env = EnvWithGoal(create_maze_env(ENV_NAME), ENV_NAME)

        state_dim = env.state_space.shape[0]
        action_dim = env.action_space.shape[0]

        obs = env.reset()
        pos = obs['achieved_goal']
        position_dim = pos.shape[0]

        # {xyz=3, orientation (quaternion)=4, limb angles=8} * {pos, vel}
        # = (3+4+8)*2 = 15*2 = 30
        self.assertEqual(state_dim, 30)
        # num of limbs
        self.assertEqual(action_dim, 8)
        # Evaluate error with just xy coordinates
        self.assertEqual(position_dim, 2)

    def test_max_action(self):
        env = EnvWithGoal(create_maze_env(ENV_NAME), ENV_NAME)
        max_action = env.action_space.high
        self.assertEquals(max_action, 30)

    def test_goal_does_not_change(self):
        env = EnvWithGoal(create_maze_env(ENV_NAME), ENV_NAME)

        obs = env.reset()
        goal = obs['desired_goal']

        for i in range(100):
            a = np.random.rand(action_dim)
            obs, reward, done, info = env.step(a)
            g = obs['desired_goal']

            self.assertEqual(goal, g)

    def test_state_does_change(self):
        env = EnvWithGoal(create_maze_env(ENV_NAME), ENV_NAME)
        max_action = float(env.action_space.high[0])

        obs = env.reset()
        state = obs['observation']

        for i in range(100):
            a = np.random.rand(action_dim)
            a = np.clip(max_action*a, -max_action, max_action)
            obs, reward, done, info = env.step(a)
            s = obs['observation']

            self.assertNotEqual(state, s)

    def test_reward_equation(self):
        env = EnvWithGoal(create_maze_env(ENV_NAME), ENV_NAME)

        obs = env.reset()
        goal = obs['desired_goal']
        state = obs['observation']

        a = np.random.rand(action_dim)
        obs, reward, done, info = env.step(a)

        goal = obs['desired_goal']
        state = obs['observation']

        diff = state[:2] - goal
        squared = np.square(diff)
        sum_squared = np.sum(squared)
        mse = np.sqrt(sum_squared)
        hand_computed_reward = -mse

        self.assertEqual(reward, hand_computed_reward)


    def test_goal_range(self):
        env = EnvWithGoal(create_maze_env(ENV_NAME), ENV_NAME)

        obs = env.reset()
        goal = obs['desired_goal']

        goals = np.zeros((1000, goal.shape[0]))

        for i in range(1000):
            obs = env.reset()
            goal = obs['desired_goal']
            goals[i,:] = goal

        self.assertAlmostEqual(np.min(goal[:,0]), -4)
        self.assertAlmostEqual(np.min(goal[:,1]), -4)
        self.assertAlmostEqual(np.max(goal[:,0]), 20)
        self.assertAlmostEqual(np.max(goal[:,1]), 20)

if __name__ == '__main__':
    unittest.main(verbosity=2)
