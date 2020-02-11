import gym
from gym.spaces import MultiDiscrete

import numpy as np

import random
import sys
import time

def _clamp(val, min_val, max_val):
    if val < min_val:
        return min_val
    elif max_val < val:
        return max_val
    else:
        return val

class RiskyPathEnv(gym.Env):
    """Gridworld designed to test basic understanding of causality under stochasticity.

    This environment is proposed in section 6.3 of B. Ziebart's PhD thesis (2010).

    The objective is to start in the top left corner and reach the lower bottom corner:

    START
     |
     v
     -.....
     -****.
     ----*.
     ----*.
     ----*.
     -----_ <--GOAL

    The actions are moving RIGHT, UP, LEFT and DOWN, and all tiles are walkable.
    The reward is given by your current tile, as follows:

    '*' : -100
    '-' : -10
    '.' : -1
    '_' : 0

    If the environment is deterministic, then the best strategy is to go through the
    narrow path in the top right region; however, if we have noise in the movements
    (i.e. 50% you move the direction you chose, %50 chance you choose an adjacent
    direction), there is a chance of hitting the high cost region - because of this,
    the best strategy is then to go through the left bottom region.
    An agent that doesn't take into account the fact that it cannot choose its own
    trajectory might try to take the riskier path. 

    This is set up with a fixed episode length of 50, but movement stops once you
    reach the goal.
    """

    def __init__(self):
        self._width = 6
        self._height = 6

        self.observation_space = MultiDiscrete((self._height, self._width))
        self.action_space = Discrete(4)

        self._pos = None
        self.max_episode_steps = 50

        self._cost_values = {
                'NONE' : 0,
                'LOW' : 1,
                'MEDIUM' : 10,
                'HIGH' : 100,
                }

        self._pos_costs = np.full((self._height, self._width), 'MEDIUM')
        self._pos_costs[0, 1:] = 'LOW'
        self._pos_costs[:-1, -1] = 'LOW'

        self._pos_costs[1, 1:-1] = 'HIGH'
        self._pos_costs[1:-1, -2] = 'HIGH'

        self._pos_costs[-1, -1] = 'NONE'

        self._goal = (self._height - 1, self._width - 1)


    def reset(self):
        self.t = 0
        self._pos = (0, 0)
        return self._pos

    def step(self, act):
        assert act in self.action_space

        self._pos = self._move(act)
        reward = self._get_position_reward()
        
        self.t += 1
        done = self.t >= self.max_episode_steps
        info = {}
        return self._pos, reward, done, info

    def _move(self, act):
        if self._pos == self._goal:
            return self._pos

        moves = [
                (1, 0),
                (0, -1),
                (-1, 0),
                (0, 1),
                ]

        # 50% chance straight, 25% left, 25% right
        if random.random() < 0.5:
            final_act = act
        else:
            if random.random() < 0.5:
                final_act = (act - 1) % len(moves)
            else:
                final_act = (act + 1) % len(moves)

        dx, dy = moves[final_act]
        x, y = self._pos
        x += dx
        y += dy
        x = _clamp(x, 0, self._width - 1)
        y = _clamp(y, 0, self._height - 1)
        return (x, y)

    def _get_position_reward(self):
        return self._cost_values[self._pos_costs[self._pos]]

    def render(self, mode='human', header=True):
        cost_to_char = {
                'NONE' : '_',
                'LOW' : '.',
                'MEDIUM' : '-',
                'HIGH' : '*',
                }

        out = np.full((self._height, self._width), ' ')

        for y, row in enumerate(self._pos_costs):
            for x, cost in enumerate(row):
                out[y][x] = cost_to_char[cost]

        x, y = self._pos
        out[y][x] = '@'

        out_str = ''
        if header:
            out_str += f'Step {self.t}/{self.max_episode_steps}\n'
        
        for line in out:
            out_str += ''.join(line)
            out_str += '\n'
        out_str += '\n'

        if mode == 'human':
            print(out_str, end='')
        else:
            return out_str
