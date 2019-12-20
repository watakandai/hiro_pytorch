import unittest
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from hiro.models import TD3Actor, TD3, get_tensor

class ModelsTest(unittest.TestCase):
    def test_td3actor_output_size(self):
        max_action = get_tensor(np.random.randint(5, size=5))
        actor = TD3Actor(10, 3, 5, max_action)
        state = get_tensor(np.random.rand(10))
        goal = get_tensor(np.random.rand(3))
        y = actor(state, goal)

        self.assertEqual(y.shape[1], 5)

    def test_td3actor_output_type(self):
        max_action = get_tensor(np.random.randint(5, size=5))
        actor = TD3Actor(10, 3, 5, max_action)
        state = get_tensor(np.random.rand(10))
        goal = get_tensor(np.random.rand(3))
        y = actor(state, goal)

        self.assertEqual(type(y), torch.Tensor)

    def test_td3actor_output_minmax(self):
        random_value = 100
        sdim = 10
        gdim = 3
        adim = 5
        max_action = get_tensor(np.random.randint(random_value, size=adim))
        actor = TD3Actor(sdim, gdim, adim, max_action)

        x = np.inf * np.ones(adim)
        x = np.array([x, -x])
        x = torch.tensor(x)
        out = actor.scale * actor.max_action * torch.tanh(x)

        for i in range(adim):
            self.assertEqual(torch.max(out[0,i]), max_action[0,i])
            self.assertEqual(torch.min(out[1,i]), -max_action[0,i])


if __name__ == '__main__':
    unittest.main(verbosity=2)
