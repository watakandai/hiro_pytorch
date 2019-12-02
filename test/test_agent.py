import unittest
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from hiro.models import HiroAgent

ENV_NAME = 'AntMaze'

class AgentTest(unittest.TestCase):
    def test_(self):
        env = EnvWithGoal(create_maze_env(ENV_NAME), ENV_NAME)
        agent = HiroAgent(env)

        self.assertEqual()

    def test_low_goal_dim(self):
        env = EnvWithGoal(create_maze_env(ENV_NAME), ENV_NAME)
        agent = HiroAgent(env)

        self.assertEqual(agent.low_goal_dim, 15)

    def test_low_reward(self):
        env = EnvWithGoal(create_maze_env(ENV_NAME), ENV_NAME)
        agent = HiroAgent(env)

        goal = np.array([5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 , 0])

        state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        next_state = np.array([1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        reward1 = agent.low_reward(state, goal, next_state)

        state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        next_state = np.array([1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        reward2 = agent.low_reward(state, goal, next_state)

        self.assertTrue(reward2 > reward1)

    def test_low_reward_negative(self):
        env = EnvWithGoal(create_maze_env(ENV_NAME), ENV_NAME)
        agent = HiroAgent(env)

        goal = np.array([5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 , 0])

        state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        next_state = np.array([1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        reward1 = agent.low_reward(state, goal, next_state)

        state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        next_state = np.array([-1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        reward2 = agent.low_reward(state, goal, next_state)

        self.assertTrue(reward1 > reward2)

    def test_subgoal_transition(self):
        env = EnvWithGoal(create_maze_env(ENV_NAME), ENV_NAME)
        agent = HiroAgent(env)

        goal = np.array([5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 , 0])

        state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        next_state = np.array([1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        subgoal = agent.subgoal_transition(state, goal, next_state)

        # distance from current state to current goal should be maintained
        self.assertEqual(goal-state, subgoal-next_state)


if __name__ == '__main__':
    unittest.main(verbosity=2)
