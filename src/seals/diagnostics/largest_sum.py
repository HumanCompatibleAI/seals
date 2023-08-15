"""Environment testing scalability to high-dimensional tasks."""

from gymnasium import spaces
import numpy as np

from seals import base_envs


class LargestSumEnv(base_envs.ResettableMDP):
    """High-dimensional linear classification problem.

    This environment evaluates how algorithms scale with increasing
    dimensionality.  It is a classification task with binary actions
    and uniformly sampled states s in [0, 1]**L.  The agent is
    rewarded for taking action 1 if the sum of the first half x[:L//2]
    is greater than the sum of the second half x[L//2:], and otherwise
    is rewarded for taking action 0.
    """

    def __init__(self, length: int = 50):
        """Build environment.

        Args:
            length: dimensionality of state space vector.
        """
        super().__init__()
        self._length = length
        self.state_space = spaces.Box(low=0.0, high=1.0, shape=(length,))
        self.action_space = spaces.Discrete(2)

    def terminal(self, state: np.ndarray, n_actions_taken: int) -> bool:
        """Always returns True, since this task should have a 1-timestep horizon."""
        return True

    def initial_state(self) -> np.ndarray:
        """Returns vector sampled uniformly in [0, 1]**L."""
        init_state = self.np_random.random((self._length,))
        return init_state.astype(self.observation_space.dtype)

    def reward(self, state: np.ndarray, act: int, next_state: np.ndarray) -> float:
        """Returns +1.0 reward when action is the right label and 0.0 otherwise."""
        n = self._length
        label = np.sum(state[: n // 2]) > np.sum(state[n // 2 :])
        return float(act == label)

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        """Returns same state."""
        return state
