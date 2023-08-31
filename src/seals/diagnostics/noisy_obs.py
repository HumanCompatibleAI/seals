"""Environment testing for robustness to noise."""

from gymnasium import spaces
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

    def __init__(self, *, size: int = 5, noise_length: int = 20):
        """Build environment.

        Args:
            size: width and height of gridworld.
            noise_length: dimension of noise vector in observation.
        """
        super().__init__()

        self._size = size
        self._noise_length = noise_length
        self._goal = np.array([self._size // 2, self._size // 2])

        obs_box_low = np.concatenate(
            ([0, 0], np.full(self._noise_length, -np.inf)),  # type: ignore
        )
        obs_box_high = np.concatenate(
            ([size - 1, size - 1], np.full(self._noise_length, np.inf)),  # type: ignore
        )

        self.state_space = spaces.MultiDiscrete([size, size])
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=obs_box_low,
            high=obs_box_high,
            dtype=np.float32,
        )

    def terminal(self, state: np.ndarray, n_actions_taken: int) -> bool:
        """Always returns False."""
        return False

    def initial_state(self) -> np.ndarray:
        """Returns one of the grid's corners."""
        n = self._size
        corners = np.array([[0, 0], [n - 1, 0], [0, n - 1], [n - 1, n - 1]])
        return corners[self.np_random.integers(4)]

    def reward(self, state: np.ndarray, action: int, new_state: np.ndarray) -> float:
        """Returns  +1.0 reward if state is the goal and 0.0 otherwise."""
        return float(np.all(state == self._goal))

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        """Returns next state according to grid."""
        return util.grid_transition_fn(
            state,
            action,
            x_bounds=(0, self._size - 1),
            y_bounds=(0, self._size - 1),
        )

    def obs_from_state(self, state: np.ndarray) -> np.ndarray:
        """Returns (x, y) concatenated with Gaussian noise."""
        noise_vector = self.np_random.normal(size=self._noise_length)
        return np.concatenate([state, noise_vector]).astype(np.float32)
