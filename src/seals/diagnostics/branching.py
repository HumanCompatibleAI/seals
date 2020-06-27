"""Hard-exploration environment."""

import itertools

import numpy as np

from seals import base_envs


class BranchingEnv(base_envs.TabularModelPOMDP):
    """Long branching environment requiring exploration.

    The agent must traverse a specific path of length L to reach a
    final goal, with B choices at each step. Wrong actions lead to
    dead-ends with zero reward.
    """

    def __init__(self, branch_factor: int = 2, length: int = 10):
        """Construct environment.

        Args:
            branch_factor: number of actions at each state.
            length: path length from initial state to goal.
        """
        self._branch_factor = branch_factor

        nS = 1 + branch_factor * length
        self._n_states = nS
        nA = branch_factor

        transition_matrix = np.zeros((nS, nA, nS))
        for ob, act in itertools.product(range(nS), range(nA)):
            transition_matrix[ob, act, self._get_next(ob, act)] = 1.0

        reward_matrix = np.zeros((nS,))
        reward_matrix[-1] = 1.0

        super().__init__(
            transition_matrix=transition_matrix, reward_matrix=reward_matrix,
        )

    def _get_next(self, state: int, action: int) -> int:
        b = self._branch_factor
        n = self._n_states
        return state + (action + 1) * int(state % b == 0 and state != n - 1)
