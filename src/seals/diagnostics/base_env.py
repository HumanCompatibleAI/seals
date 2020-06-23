"""Minimal base class for simple environments."""

import gym
from gym.spaces import Discrete
from gym.utils import seeding
import numpy as np

from seals.util import sample_distribution


class BaseEnv(gym.Env):
    """Minimal envrironment class, with default tabular method implementations."""

    def __init__(self, num_states=None, num_actions=None):
        super().__init__()

        if num_states is not None:
            self.observation_space = Discrete(num_states)
        if num_actions is not None:
            self.action_space = Discrete(num_actions)

        self.seed()

    def seed(self, seed=None):
        """Set randomness seed."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Start a new episode, returning initial observation."""
        self.state = self.sample_initial_state()
        return self.ob_from_state(self.state)

    def step(self, action):
        """Advance to next timestep using provided action."""
        assert action in self.action_space, f"{action} not in {self.action_space}"

        old_state = self.state
        self.state = self.transition_fn(self.state, action)
        next_ob = self.ob_from_state(self.state)

        reward = self.reward_fn(old_state, action, self.state)

        done = self.termination_fn(self.state)
        info = {}

        return next_ob, reward, done, info

    def reward_fn(self, state, action, new_state):
        """Return reward value R(s, a, s')."""
        return self.reward_matrix[state, action, new_state]

    def transition_fn(self, state, action):
        """Sample next state according to T(s, a)."""
        return sample_distribution(
            self.transition_matrix[state, action], random=self.np_random
        )

    def ob_from_state(self, state):
        """Return observation from current state."""
        return state

    def state_from_ob(self, ob):
        """Return state from current observation, when possible."""
        return ob

    def sample_initial_state(self):
        """Sample initial state.

        Default implementation assumes agent starts at
        state given by zero entries.
        """
        return np.zeros(
            self.observation_space.shape, dtype=self.observation_space.dtype
        )

    def initial_state_distribution(self):
        """Get initial state distribution

        Default implementation assumes initial state is
        deterministic, sampling an initial state and returning
        the deterministic distribution corresponding to the
        sampled state.
        """
        initial_state = self.sample_initial_state()
        nS = self.observation_space.n
        one_hot_state = np.eye(nS)[initial_state]
        return one_hot_state

    def termination_fn(self, state):
        """Returns whether state is terminal.

        Default implementation assumes state is never terminal.
        """
        return False

    def render(self):
        """Render environment state."""
        print(self.state)
