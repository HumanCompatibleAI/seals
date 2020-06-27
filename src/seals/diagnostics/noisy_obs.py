"""Environment testing for robustness to noise."""

from gym import spaces
import numpy as np

from seals import base_envs, util


class NoisyObsEnv(base_envs.ResettablePOMDP):
    """Simple gridworld with noisy observations.

    The agent randomly starts at the one of the corners of an MxM grid and
    tries to reach and stay at the center. The observation consists of the
    agent's (x,y) coordinates and L "distractor" samples of Gaussian noise .
    The challenge is to select the relevant features in the observations, and
    not overfit to noise.
    """
    def __init__(self, *, size:int = 5, noise_length:int = 20):
        """Build environment.

        Args:
            size: width and height of gridworld.
            noise_length: dimension of noise vector in observation.
        """
        self._size = size
        self._noise_length = noise_length
        self._goal = np.array([self._size // 2, self._size // 2])

        self._observation_space = spaces.Box(
            low=np.concatenate(([0, 0], np.full(self._noise_length, -np.inf),)),
            high=np.concatenate(
                ([size - 1, size - 1], np.full(self._noise_length, np.inf),)
            ),
            dtype=float,
        )

        super().__init__(
            state_space=spaces.MultiDiscrete([size, size]),
            action_space=spaces.Discrete(5),
        )

    def terminal(self, state: int) -> bool:
        return False

    def initial_state(self) -> int:
        n = self._size
        corners = np.array([[0, 0], [n - 1, 0], [0, n - 1], [n - 1, n - 1]])
        return corners[np.random.randint(4)]

    def reward(self, state: int, action: int, new_state: int) -> float:
        return np.allclose(state, self.goal)

    def transition(self, state: int, action: int) -> int:
        return util.grid_transition_fn(
            state, action, x_bounds=(0, self._size - 1), y_bounds=(0, self._size - 1)
        )

    @property
    def observation_space(self):
        return self._observation_space

    def ob_from_state(self, state):
        noise_vector = self.np_random.randn(self._noise_length)
        return np.concatenate([state, noise_vector])
