"""Environment to sort a list using swap actions."""

from gymnasium import spaces
import numpy as np

from seals import base_envs


class SortEnv(base_envs.ResettableMDP):
    """Environment to sort a list using swap actions."""

    def __init__(self, length: int = 4):
        """Constructs environment.

        The initial state is a vector x sampled uniformly from
        [0,1]**L, with actions a = (i,j) swapping x_i and x_j.  The
        reward is given according to the number of elements in the
        correct position.  To perform well, the learned policy must
        compare elements, otherwise it will not generalize to all
        possible randomly selected initial states.
        """
        self._length = length

        super().__init__()
        self.state_space = spaces.Box(low=0, high=1.0, shape=(length,))
        self.action_space = spaces.MultiDiscrete([length, length])

    def terminal(self, state: np.ndarray, n_actions_taken: int) -> bool:
        """Always returns False."""
        return False

    def initial_state(self):
        """Sample random vector uniformly in [0, 1]**L."""
        sample = self.np_random.random(size=self._length)
        return sample.astype(self.state_space.dtype)

    def reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        new_state: np.ndarray,
    ) -> float:
        """Rewards fully sorted lists, and new correct positions."""
        del action
        # This is not meant to be a potential shaping in the formal sense,
        # as it changes the trajectory returns (since we do not return
        # a fixed-potential state at termination).
        num_correct = self._num_correct_positions(state)
        new_num_correct = self._num_correct_positions(new_state)
        potential_diff = new_num_correct - num_correct

        return float(self._is_sorted(new_state)) + potential_diff

    def transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Action a = (i, j) swaps elements in positions i and j."""
        new_state = state.copy()
        i, j = action
        new_state[[i, j]] = new_state[[j, i]]
        return new_state

    def _is_sorted(self, arr: np.ndarray) -> bool:
        return list(arr) == sorted(arr)

    def _num_correct_positions(self, arr: np.ndarray) -> int:
        return np.sum(arr == sorted(arr))
